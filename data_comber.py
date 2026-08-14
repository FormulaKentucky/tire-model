import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functools import reduce
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import least_squares
import scipy.io

def sort(data, load_key, window, threshold_factor):
    # Extract and ensure the load signal is 1‑D
    load = np.asarray(data[load_key]).squeeze()   # <--- fix: squeeze to 1D
    if load.ndim != 1:
        raise ValueError(f"Vertical load channel must be 1‑D after squeezing, "
                         f"but got shape {load.shape}.")

    n = len(load)

    # ----- rolling variance -----
    windows = sliding_window_view(load, window)          # shape (n-window+1, window)
    rolling_var = np.var(windows, axis=1, ddof=0)       # population variance

    # Centre the variance values: window i is assigned to index i + window//2
    var_centered = np.full(n, np.nan)
    centre_offset = window // 2
    var_centered[centre_offset : centre_offset + len(rolling_var)] = rolling_var

    # ----- threshold -----
    valid = rolling_var[np.isfinite(rolling_var)]
    thresh = threshold_factor * np.median(valid)

    # ----- constant‑load mask -----
    is_constant = var_centered < thresh   # NaN -> False

    # ----- find contiguous constant blocks -----
    block_starts = []
    block_ends = []
    in_block = False
    for i in range(n):
        if is_constant[i] and not in_block:
            in_block = True
            block_starts.append(i)
        elif not is_constant[i] and in_block:
            in_block = False
            block_ends.append(i - 1)
    if in_block:
        block_ends.append(n - 1)

    # ----- slice the original data -----
    segmented = []
    for start, end in zip(block_starts, block_ends):
        seg = {}
        for key, arr in data.items():
            seg[key] = arr[start:end + 1]
        segmented.append(seg)

    return segmented

def bound(
    segments,
    slip_key='SA',
    window=50,
    threshold=None,
    threshold_factor=2.0,
    margin=0,
    min_sweep_length=10
):
    trimmed = []

    for seg in segments:
        # --- safe 1‑D extraction (fixes the 0‑d issue) ---
        sa = np.atleast_1d(np.asarray(seg[slip_key])).ravel()
        n = len(sa)

        # Edge case: too few points for rolling average
        if n < window:
            # Fall back: use the whole segment as-is
            trimmed.append(seg)
            print(f"Warning: segment too short ({n} samples) for window {window}; "
                  f"returned untrimmed.")
            continue

        # --- rolling average (centred) ---
        boxcar = np.ones(window) / window
        rm = np.convolve(sa, boxcar, mode='same')
        half = window // 2
        rm[:half] = rm[half]
        rm[-half:] = rm[-half - 1]

        # --- baseline from initial steady part ---
        baseline = np.median(rm[:window])

        deviation = np.abs(rm - baseline)

        # --- threshold ---
        if threshold is None:
            th = threshold_factor * np.std(rm)
        else:
            th = threshold

        # --- find sweep boundaries ---
        high_dev = deviation > th
        high_idx = np.where(high_dev)[0]

        if len(high_idx) < 2:
            print(f"Warning: no sweep detected in a segment; leaving it untrimmed.")
            trimmed.append(seg)
            continue

        start_raw = high_idx[0]
        end_raw = high_idx[-1]

        start = max(0, start_raw - margin)
        end = min(n - 1, end_raw + margin)

        if (end - start + 1) < min_sweep_length:
            print(f"Warning: detected sweep too short "
                  f"({end - start + 1} samples); leaving segment untrimmed.")
            trimmed.append(seg)
            continue

        # --- slice all channels ---
        trimmed_seg = {}
        for key, arr in seg.items():
            trimmed_seg[key] = arr[start:end + 1]
        trimmed.append(trimmed_seg)

    return trimmed

def bound2(
    segments,
    fy_key='FY',
    window=50,
    threshold_factor=5,
    min_steady_length=10,
    margin=0
):
    trimmed = []

    for seg in segments:
        # Extract lateral force and guarantee 1-D
        fy = np.atleast_1d(np.asarray(seg[fy_key])).ravel()
        n = len(fy)

        # --- rolling variance (centred) ---
        # Build sliding windows and compute variance
        # We'll use a loop for clarity (or convolution of squared signal minus mean squared).
        # Using sliding_window_view if available, else manual.

        # Compute rolling mean first
        boxcar = np.ones(window) / window
        mean_fy = np.convolve(fy, boxcar, mode='same')
        # Rolling mean of squared values
        mean_sq = np.convolve(fy**2, boxcar, mode='same')
        # Variance = E[X^2] - (E[X])^2
        roll_var = mean_sq - mean_fy**2

        # Fix edge effects: repeat the first/last valid value
        half = window // 2
        roll_var[:half] = roll_var[half]
        roll_var[-half:] = roll_var[-half - 1]

        # Ensure no negative values due to floating‑point errors
        roll_var = np.maximum(roll_var, 0.0)

        # --- threshold ---
        # Use median of valid (non‑NaN) variance as baseline noise floor
        valid_var = roll_var[np.isfinite(roll_var)]
        thresh = threshold_factor * np.median(valid_var)

        # --- steady‑state mask ---
        steady = roll_var < thresh

        # --- find contiguous steady blocks ---
        block_starts = []
        block_ends = []
        in_block = False
        for i in range(n):
            if steady[i] and not in_block:
                in_block = True
                block_starts.append(i)
            elif not steady[i] and in_block:
                in_block = False
                block_ends.append(i - 1)
        if in_block:
            block_ends.append(n - 1)

        if not block_starts:
            print("Warning: no steady FY region found; segment left untrimmed.")
            trimmed.append(seg)
            continue

        # Pick the longest block
        lengths = [e - s + 1 for s, e in zip(block_starts, block_ends)]
        idx = np.argmax(lengths)
        start_raw = block_starts[idx]
        end_raw = block_ends[idx]

        # Apply margin, clipped
        start = max(0, start_raw - margin)
        end = min(n - 1, end_raw + margin)

        if (end - start + 1) < min_steady_length:
            print(f"Warning: longest steady block too short "
                  f"({end - start + 1} samples); segment left untrimmed.")
            trimmed.append(seg)
            continue

        # --- slice all channels ---
        trimmed_seg = {}
        for key, arr in seg.items():
            trimmed_seg[key] = arr[start:end + 1]
        trimmed.append(trimmed_seg)

    return trimmed

cases = []

# split9 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw4.mat'), load_key="FZ", window=200, threshold_factor=30)

# # for i in range(len(split9)):
# #     plt.plot(split9[i]["ET"],split9[i]["FZ"])

# split = bound(split9, slip_key="SA", threshold_factor=0.86)

# # for i in range(len(split)):
# #     plt.plot(split[i]["ET"],split[i]["SA"])

# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>8):
#         cases.append(split[i])
#         plt.plot(split[i]["SA"],split[i]["MZ"])
        

split9_69 = sort(scipy.io.loadmat('data/straight_SI/B2356raw69.mat'), load_key="FZ", window=100, threshold_factor=10)
split = bound(split9_69, slip_key="SL", threshold_factor=1)
for i in range(len(split)):
    if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1) & (np.abs(split[i]["IA"]).mean() < 1e-1):
        cases.append(split[i])
        plt.plot(split[i]["ET"],split[i]["FZ"])
