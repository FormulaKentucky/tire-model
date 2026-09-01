import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from functools import reduce
from scipy.optimize import least_squares, minimize
from scipy.special import huber
import scipy.io

# Use a rolling variance to detmerine the boundaries for loading cases
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

# Use a rolling average to determine the bounds for the slip angle
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

# Find the bounds for trimming based on rolling variance
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



#-------------------------------------------------------------------------------------------------
#                       LATERAL FORCE MAGIC FORMULA FITTING
#-------------------------------------------------------------------------------------------------



def first_pass_y(data, x): 
    # Get the data
    F_y = -data["FY"]
    alpha = np.tan(data["SA"] * np.pi / 180)
        
    # Get the parameters
    B_y = x[0]
    C_y = x[1]
    D_y = x[2]
    E_y = x[3]
    S_hy = x[4]
    S_vy = x[5]
        
    # FORGE THE MAGIC FORMULA
    Y = D_y * np.sin(C_y * np.arctan(B_y * (alpha + S_hy) - E_y * (B_y * (alpha + S_hy) - np.arctan(B_y * (alpha + S_hy))))) + S_vy
    
    return (Y - F_y).squeeze()

def second_pass_y(data, F_z0, lambda_mu_y, BCDE_params, x):    
    # Get the BCDE parameters
    B_y = BCDE_params[:,0]
    C_y = BCDE_params[:,1]
    D_y = BCDE_params[:,2]
    E_y = BCDE_params[:,3]
    S_hy = BCDE_params[:,4]
    S_vy = BCDE_params[:,5]
        
    # Get the P parameters (THIS IS THE ORDER OF THE OUTPUT PARAMETERS)
    PDY1 = x[0]
    PDY2 = x[1]
    PDY3 = x[2]
    PCY1 = x[3]
    PKY1 = x[4]
    PKY2 = x[5]
    PKY3 = x[6]
    PKY4 = x[7]
    PKY5 = x[8]
    PKY6 = x[9]
    PKY7 = x[10]
    PHY1 = x[11]
    PHY2 = x[12]
    PEY1 = x[13]
    PEY2 = x[14]
    PEY3 = x[15]
    PEY4 = x[16]
    PEY5 = x[17]
    PVY1 = x[18]
    PVY2 = x[19]
    PVY3 = x[20]
    PVY4 = x[21]
        
    # Initialize arrays
    F_z = -data[0]["FZ"]
    alpha = np.tan(data[0]["SA"] * np.pi / 180)
    gamma = np.sin(data[0]["IA"] * np.pi / 180)
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
        
    # Solve for some stuff before hand
    mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2) 
    BCD_y = PKY1 * F_z0 * np.sin(PKY4 * np.arctan(F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0))) / (1 + PKY3 * gamma ** 2)
    S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
    K_y_gamma_0 = F_z * (PKY6 + PKY7 * df_z)
        
    # Fit parameters to B C D E P_hy and P_vy
    shit = (
    D_y[0] - (mu_y * F_z),
    C_y[0] - PCY1,
    B_y[0] - BCD_y / (mu_y * F_z * PCY1 + 1e-8),
    S_hy[0] - ((PHY1 + PHY2 * df_z) + (K_y_gamma_0 * gamma - S_vy_gamma) / (BCD_y + 1e-8)),
    E_y[0] - ((PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) * np.sign(alpha + ((PHY1 + PHY2 * df_z) + (K_y_gamma_0 * gamma - ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma)) / (BCD_y + 1e-8))))),
    S_vy[0] - ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma)
    )
    
    # print((D_y[0] - (mu_y * F_z)).shape)
    # print((C_y[0] - PCY1).shape)
    # print((B_y[0] - BCD_y / (mu_y * F_z * PCY1)).shape)
    # print((S_hy[0] - (PHY1 + PHY2 * df_z)).shape)
    # print((E_y[0] - ((PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) - np.sign(alpha + S_hy)))).shape)
    # print((S_vy[0] - ((PVY1 + PVY2 * df_z) * F_z)).shape)
    # print(gamma.shape)
    
    
    residuals = np.vstack(shit)
    
    for i in range(1,len(data)):
        # Initialize arrays
        F_z = -data[i]["FZ"]
        alpha = np.tan(data[i]["SA"] * np.pi / 180)
        gamma = np.sin(data[i]["IA"] * np.pi / 180)
        
        # Load sensitivity factor
        df_z = (F_z - F_z0) / F_z0
            
        # Solve for some stuff before hand
        mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2) 
        BCD_y = PKY1 * F_z0 * np.sin(PKY4 * np.arctan(F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0))) / (1 + PKY3 * gamma ** 2)
        S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
        K_y_gamma_0 = F_z * (PKY6 + PKY7 * df_z)
            
        # Fit parameters to B C D E P_hy and P_vy
        shit = (
        residuals,
        D_y[i] - (mu_y * F_z),
        C_y[i] - PCY1,
        B_y[i] - BCD_y / (mu_y * F_z * PCY1 + 1e-8),
        S_hy[i] - ((PHY1 + PHY2 * df_z) + (K_y_gamma_0 * gamma - S_vy_gamma) / (BCD_y + 1e-8)),
        E_y[i] - ((PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) * np.sign(alpha + ((PHY1 + PHY2 * df_z) + (K_y_gamma_0 * gamma - ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma)) / (BCD_y + 1e-8))))),
        S_vy[i] - ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma)
        )
        
        
        residuals = np.vstack(shit)
        
    # print(residuals.size)
    # print(residuals.shape)
    
    return residuals.squeeze()



# This is the data for 16.0X7.5-10, R20, C2000; Rim_Width=7.0
F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/cornering_SI/B2356raw4.mat')["FZ"],
                         scipy.io.loadmat('data/cornering_SI/B2356raw5.mat')["FZ"],
                         scipy.io.loadmat('data/cornering_SI/B2356raw6.mat')["FZ"]))).mean()
cases = []

# # Split up the data for the cloudy shit
# split9_2 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw2.mat'), load_key="SA", window=80, threshold_factor=10)
# split = bound2(split9_2, window=110, threshold_factor=5)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<7) & (split[i]["ET"].max()-split[i]["ET"].min()>3):
#         cases.append(split[i])

# Split up the data for the continuous sweeps                  
split9_4 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw4.mat'), load_key="FZ", window=200, threshold_factor=30)
split = bound(split9_4, threshold_factor=0.86)
for i in range(len(split)):
    if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>8):
        cases.append(split[i])
        
        
split9_5 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw5.mat'), load_key="FZ", window=250, threshold_factor=60)
split = bound(split9_5, threshold_factor=1)
for i in range(len(split)):
    if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>8):
        cases.append(split[i])
        
        
split9_6 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw6.mat'), load_key="FZ", window=250, threshold_factor=50)
split = bound(split9_6, threshold_factor=1)
for i in range(len(split)):
    if (split[i]["ET"].max()-split[i]["ET"].min()<13) & (split[i]["ET"].max()-split[i]["ET"].min()>10):
        cases.append(split[i])

BCDE_params = np.zeros((len(cases),6))

for i in range(len(cases)):
    if i == 0:
        x0_BCDE = [0, 1.45, 500, 0, 0, 0]
    else:
        x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
    fit_func = lambda x:(first_pass_y(cases[i],x))
    result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
                           bounds=([0, 1, 0, 0, -0.075, -100],
                                   [30, 2, 100 + np.abs(cases[i]["FY"]).max(), 1, 0.075, 100]),
                           # x_scale="jac",
                           # diff_step=1e-4,
                           ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
                           max_nfev=int(1e+8), verbose=1)
    
    BCDE_params[i] = result.x

x0_P = [1, 0, 0, BCDE_params[:,1].mean(), 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0.15, 0]


fit_func = lambda x:(second_pass_y(cases, F_z0, 1, BCDE_params, x))
result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
                       # x_scale="jac",
                       # diff_step=1e-4,
                       ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
                       max_nfev=int(1e+8), verbose=1)

lat_160X75_R20_70 = np.hstack((result.x, 16.0, F_z0)) 



# # This is for Hoosier 16.0X7.5-10, R20, C2000; Rim_Width=8.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/cornering_SI/B2356raw8.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw7.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw9.mat')["FZ"]))).mean()
# cases = []

# # Split up the data for the continuous sweeps
# split9_8 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw8.mat'), load_key="FZ", window=130, threshold_factor=90)
# split = bound(split9_8, threshold_factor=0.86)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>6):
#         cases.append(split[i])
        
# split9_9 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw9.mat'), load_key="FZ", window=130, threshold_factor=100)
# split = bound(split9_9, threshold_factor=0.86)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>6):
#         cases.append(split[i])

# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_y(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, 0, -0.075, -100],
#                                    [30, 2, 100 + np.abs(cases[i]["FY"]).max(), 1, 0.075, 100]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, 0, BCDE_params[:,1].mean(), 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0.15, 0]

# fit_func = lambda x:(second_pass_y(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# lat_160X75_R20_80 = np.hstack((result.x, 16.0, F_z0))



# # This is the data for 20.5X7.0-13, R20; Rim_Width=7.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/cornering_SI/B2356raw16.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw17.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw18.mat')["FZ"]))).mean()
# cases = []

# # Split up the data for the continuous sweeps                  
# split9_17 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw17.mat'), load_key="FZ", window=130, threshold_factor=120)
# split = bound(split9_17, threshold_factor=0.86)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>6):
#         cases.append(split[i])     
        
# split9_18 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw18.mat'), load_key="FZ", window=130, threshold_factor=90)
# split = bound(split9_18, threshold_factor=0.86)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>6):
#         cases.append(split[i])

# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_y(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, 0, -0.075, -100],
#                                    [30, 2, 100 + np.abs(cases[i]["FY"]).max(), 1, 0.075, 100]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, 0, BCDE_params[:,1].mean(), 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0.15, 0]


# fit_func = lambda x:(second_pass_y(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# lat_205X70_R20_70 = np.hstack((result.x, 20.5, F_z0)) 



# # This is the data for 20.5X7.0-13, R20; Rim_Width=8.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/cornering_SI/B2356raw19.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw20.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw21.mat')["FZ"]))).mean()
# cases = []

# # Split up the data for the continuous sweeps                  
# split9_17 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw20.mat'), load_key="FZ", window=130, threshold_factor=90)
# split = bound(split9_17, threshold_factor=0.86)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>6):
#         cases.append(split[i])     
        
# split9_18 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw21.mat'), load_key="FZ", window=130, threshold_factor=90)
# split = bound(split9_18, threshold_factor=0.86)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>6):
#         cases.append(split[i])

# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_y(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, 0, -0.075, -100],
#                                    [30, 2, 100 + np.abs(cases[i]["FY"]).max(), 1, 0.075, 100]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, 0, BCDE_params[:,1].mean(), 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0.15, 0]


# fit_func = lambda x:(second_pass_y(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# lat_205X70_R20_80 = np.hstack((result.x, 20.5, F_z0)) 



# # This is the data for 18.0X6.0-10, R20; Rim_Width=6.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/cornering_SI/B2356raw27.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw28.mat')["FZ"],
#                          scipy.io.loadmat('data/cornering_SI/B2356raw29.mat')["FZ"]))).mean()
# cases = []

# # Split up the data for the continuous sweeps                  
# split9_28 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw28.mat'), load_key="FZ", window=190, threshold_factor=20)
# split = bound(split9_28, threshold_factor=0.86)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>9):
#         cases.append(split[i])
        
# split9_29 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw29.mat'), load_key="FZ", window=110, threshold_factor=18)
# split = bound(split9_29, threshold_factor=0.89)

# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>10):
#         cases.append(split[i])

# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_y(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, 0, -0.075, -100],
#                                    [30, 2, 100 + np.abs(cases[i]["FY"]).max(), 1, 0.075, 100]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, 0, BCDE_params[:,1].mean(), 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0.15, 0]


# fit_func = lambda x:(second_pass_y(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# lat_180X60_R20_60 = np.hstack((result.x, 18.0, F_z0)) 



# This is the data for 18.0X6.0-10, R20; Rim_Width=6.0
F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/cornering_SI/B2356raw30.mat')["FZ"],
                         scipy.io.loadmat('data/cornering_SI/B2356raw31.mat')["FZ"],
                         scipy.io.loadmat('data/cornering_SI/B2356raw32.mat')["FZ"]))).mean()
cases = []

# Split up the data for the continuous sweeps                  
split9_31 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw31.mat'), load_key="FZ", window=110, threshold_factor=18)
split = bound(split9_31, threshold_factor=0.89)
for i in range(len(split)):
    if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>10):
        cases.append(split[i])
        
split9_32 = sort(scipy.io.loadmat('data/cornering_SI/B2356raw32.mat'), load_key="FZ", window=110, threshold_factor=18)
split = bound(split9_32, threshold_factor=0.89)
for i in range(len(split)):
    if (split[i]["ET"].max()-split[i]["ET"].min()<12) & (split[i]["ET"].max()-split[i]["ET"].min()>10):
        cases.append(split[i])

BCDE_params = np.zeros((len(cases),6))

for i in range(len(cases)):
    if i == 0:
        x0_BCDE = [0, 1.45, 500, 0, 0, 0]
    else:
        x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
    fit_func = lambda x:(first_pass_y(cases[i],x))
    result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
                           bounds=([0, 1, 0, 0, -0.075, -100],
                                   [30, 2, 100 + np.abs(cases[i]["FY"]).max(), 1, 0.075, 100]),
                           # x_scale="jac",
                           # diff_step=1e-4,
                           ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
                           max_nfev=int(1e+8), verbose=1)
    
    BCDE_params[i] = result.x

x0_P = [1, 0, 0, BCDE_params[:,1].mean(), 10, 1.5, 0, 2, 0, 2.5, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0.15, 0]


fit_func = lambda x:(second_pass_y(cases, F_z0, 1, BCDE_params, x))
result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
                       # x_scale="jac",
                       # diff_step=1e-4,
                       ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
                       max_nfev=int(1e+8), verbose=1)

lat_180X60_R20_70 = np.hstack((result.x, 18.0, F_z0)) 



#-------------------------------------------------------------------------------------------------
#                       LONGITUDINAL FORCE MAGIC FORMULA FITTING
#-------------------------------------------------------------------------------------------------



def first_pass_x(data, x): 
    # Get the data
    F_x = data["FX"]
    s = data["SL"]
        
    # Get the parameters
    B_x = x[0]
    C_x = x[1]
    D_x = x[2]
    E_x = x[3]
    S_hx = x[4]
    S_vx = x[5]
        
    # FORGE THE MAGIC FORMULA
    Y = D_x * np.sin(C_x * np.arctan(B_x * (s + S_hx) - E_x * (B_x * (s + S_hx) - np.arctan(B_x * (s + S_hx))))) + S_vx
    
    
    return (Y - F_x).squeeze()

 
def second_pass_x(data, F_z0, lambda_mu_x, BCDE_params, x):    
    # Get the BCDE parameters
    B_x = BCDE_params[:,0]
    C_x = BCDE_params[:,1]
    D_x = BCDE_params[:,2]
    E_x = BCDE_params[:,3]
    S_hx = BCDE_params[:,4]
    S_vx = BCDE_params[:,5]
        
    # Get the P parameters (THIS IS THE ORDER OF THE OUTPUT PARAMETERS)
    PDX1 = x[0]
    PDX2 = x[1]
    PCX1 = x[2]
    PKX1 = x[3]
    PKX2 = x[4]
    PKX3 = x[5]
    PHX1 = x[6]
    PHX2 = x[7]
    PEX1 = x[8]
    PEX2 = x[9]
    PEX3 = x[10]
    PEX4 = x[11]
    PVX1 = x[12]
    PVX2 = x[13]
        
    # Initialize arrays
    F_z = -data[0]["FZ"]
    s = data[0]["SL"]
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
        
    # Solve for some stuff before hand
    mu_x = (PDX1 + PDX2 * df_z) * lambda_mu_x
    BCD_x = F_z * (PKX1 + PKX2 * df_z) * np.exp(PKX3 * df_z)
        
    # Fit parameters to B C D E P_hy and P_vy
    shit = (
    D_x[0] - (mu_x * F_z),
    C_x[0] - PCX1,
    B_x[0] - BCD_x / (mu_x * F_z * PCX1),
    S_hx[0] - (PHX1 + PHX2 * df_z),
    E_x[0] - ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2) * (1 - PEX4 * np.sign(s + (PHX1 + PHX2 * df_z)))),
    S_vx[0] - (F_z * (PVX1 + PVX2 * df_z))
    )
    
    # print((D_x[0] - (mu_x * F_z)).shape)
    # print((C_x[0] - PCX1).shape)
    # print((B_x[0] - BCD_x / (mu_x * F_z * PCX1)).shape)
    # print((S_hx[0] - (PHX1 + PHX2 * df_z)).shape)
    # print((E_x[0] - ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2) * (1 - PEX4 * np.sign(s + S_hx)))).shape)
    # print((S_vx[0] - (F_z * (PVX1 + PVX2 * df_z))).shape)
    
    
    residuals = np.vstack(shit)
    
    for i in range(1,len(data)):
        # Initialize arrays
        F_z = -data[i]["FZ"]
        s = data[i]["SL"]
        
        # Load sensitivity factor
        df_z = (F_z - F_z0) / F_z0
            
        # Solve for some stuff before hand
        mu_x = (PDX1 + PDX2 * df_z) * lambda_mu_x
        BCD_x = F_z * (PKX1 + PKX2 * df_z) * np.exp(PKX3 * df_z)
            
        # Fit parameters to B C D E P_hy and P_vy
        shit = (
        residuals,
        D_x[i] - (mu_x * F_z),
        C_x[i] - PCX1,
        B_x[i] - BCD_x / (mu_x * F_z * PCX1 + 1e-8),
        S_hx[i] - (PHX1 + PHX2 * df_z),
        E_x[i] - ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2) * (1 - PEX4 * np.sign(s + (PHX1 + PHX2 * df_z)))),
        S_vx[i] - (F_z * (PVX1 + PVX2 * df_z))
        )
        
        
        
        residuals = np.vstack(shit)
    
    return residuals.squeeze()
    # return huber(1.35, residuals.squeeze()).sum()


# # This is the data for Hoosier 20.5X7.0-13, R20; Rim_Width=7.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/straight_SI/B2356raw51.mat')["FZ"],
#                          scipy.io.loadmat('data/straight_SI/B2356raw52.mat')["FZ"]))).mean()

# cases = []

# # Split up the data for the continuous sweeps
# split9_51 = sort(scipy.io.loadmat('data/straight_SI/B2356raw51.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_51, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])

# split9_52 = sort(scipy.io.loadmat('data/straight_SI/B2356raw52.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_52, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])
        
# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_x(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1.2, 0, 0, -0.075, -200],
#                                    [20, 2, 200 + np.abs(cases[i]["FX"]).max(), 1, 0.075, 200]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, BCDE_params[:,1].mean(), 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0]

# fit_func = lambda x:(second_pass_x(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# long_205X70_R20_70 = np.hstack((result.x, 20.5, F_z0))



# # This is the data for Hoosier 20.5X7.0-13, R20; Rim_Width=8.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/straight_SI/B2356raw54.mat')["FZ"],
#                          scipy.io.loadmat('data/straight_SI/B2356raw55.mat')["FZ"]))).mean()

# cases = []

# # Split up the data for the continuous sweeps
# split9_54 = sort(scipy.io.loadmat('data/straight_SI/B2356raw54.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_54, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])
        
# split9_55 = sort(scipy.io.loadmat('data/straight_SI/B2356raw55.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_55, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])

# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_x(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1.2, 0, 0, -0.075, -150],
#                                    [20, 2, 100 + np.abs(cases[i]["FX"]).max(), 1, 0.075, 150]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, BCDE_params[:,1].mean(), 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0]

# fit_func = lambda x:(second_pass_x(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# long_205X70_R20_80 = np.hstack((result.x, 20.5, F_z0))



# # This is the data for Hoosier 18.0X6.0-10, R20; Rim_Width=6.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/straight_SI/B2356raw69.mat')["FZ"],
#                          scipy.io.loadmat('data/straight_SI/B2356raw70.mat')["FZ"]))).mean()

# cases = []

# # Split up the data for the continuous sweeps
# split9_69 = sort(scipy.io.loadmat('data/straight_SI/B2356raw69.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_69, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])
        
# split9_70 = sort(scipy.io.loadmat('data/straight_SI/B2356raw70.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_70, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])
        
# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_x(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1.2, 0, 0, -0.075, -150],
#                                    [20, 2, 150 + np.abs(cases[i]["FX"]).max(), 1, 0.075, 150]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, BCDE_params[:,1].mean(), 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0]

# fit_func = lambda x:(second_pass_x(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# long_180X60_R20_60 = np.hstack((result.x, 18.0, F_z0))



# # This is the data for Hoosier 18.0X6.0-10, R20; Rim_Width=7.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/straight_SI/B2356raw72.mat')["FZ"],
#                          scipy.io.loadmat('data/straight_SI/B2356raw73.mat')["FZ"]))).mean()

# cases = []

# # Split up the data for the continuous sweeps
# split9_72 = sort(scipy.io.loadmat('data/straight_SI/B2356raw72.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_72, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])

# split9 = sort(scipy.io.loadmat('data/straight_SI/B2356raw73.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() < 1e-1):
#         cases.append(split[i])
        
# BCDE_params = np.zeros((len(cases),6))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCDE = [0, 1.45, 500, 0, 0, 0]
#     else:
#         x0_BCDE = [0, BCDE_params[i-1,1], 500, 0, 0, 0]
#     fit_func = lambda x:(first_pass_x(cases[i],x))
#     result = least_squares(fit_func, x0_BCDE, jac='3-point', method='trf',
#                            bounds=([0, 1.2, 0, 0, -0.075, -50],
#                                    [20, 2, 50 + np.abs(cases[i]["FX"]).max(), 1, 0.075, 50]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCDE_params[i] = result.x

# x0_P = [1, 0, BCDE_params[:,1].mean(), 12, 10, -0.6, 0, 0, -0.5, 0, 0, 0, 0, 0]

# fit_func = lambda x:(second_pass_x(cases, F_z0, 1, BCDE_params, x))
# result = least_squares(fit_func, x0_P, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# long_180X60_R20_70 = np.hstack((result.x, 18.0, F_z0))



#-------------------------------------------------------------------------------------------------
#                       NON-COMBINED F_X AND F_Y MF5.2 DEFINITIONS
#-------------------------------------------------------------------------------------------------



def tm_lat(F_z, alpha, gamma, lambda_mu_y, x):    
    # Get the P parameters (THIS IS THE ORDER OF THE OUTPUT PARAMETERS)
    PDY1 = x[0]
    PDY2 = x[1]
    PDY3 = x[2]
    PCY1 = x[3]
    PKY1 = x[4]
    PKY2 = x[5]
    PKY3 = x[6]
    PKY4 = x[7]
    PKY5 = x[8]
    PKY6 = x[9]
    PKY7 = x[10]
    PHY1 = x[11]
    PHY2 = x[12]
    PEY1 = x[13]
    PEY2 = x[14]
    PEY3 = x[15]
    PEY4 = x[16]
    PEY5 = x[17]
    PVY1 = x[18]
    PVY2 = x[19]
    PVY3 = x[20]
    PVY4 = x[21]
    F_z0 = x[-1]
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
        
    # Solve for some stuff before hand
    mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2) 
    BCD_y = PKY1 * F_z0 * np.sin(PKY4 * np.arctan(F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0))) / (1 + PKY3 * gamma ** 2)
    S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
    K_y_gamma_0 = F_z * (PKY6 + PKY7 * df_z)
        
    # Fit parameters to B C D E P_hy and P_vy
    D_y = (mu_y * F_z)
    C_y = PCY1
    B_y = BCD_y / (mu_y * F_z * PCY1)
    S_hy = ((PHY1 + PHY2 * df_z) + (K_y_gamma_0 * gamma - S_vy_gamma) / (BCD_y + 1e-8))
    E_y = ((PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) * np.sign(alpha + S_hy)))
    S_vy = ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma)
    
    # FORGE THE MAGIC FORMULA
    Y = D_y * np.sin(C_y * np.arctan(B_y * (alpha + S_hy) - E_y * (B_y * (alpha + S_hy) - np.arctan(B_y * (alpha + S_hy))))) + S_vy
    return Y, BCD_y, D_y, S_hy, S_vy, mu_y


def tm_long(F_z, s, lambda_mu_x, x):        
    # Get the P parameters (THIS IS THE ORDER OF THE OUTPUT PARAMETERS)
    PDX1 = x[0]
    PDX2 = x[1]
    PCX1 = x[2]
    PKX1 = x[3]
    PKX2 = x[4]
    PKX3 = x[5]
    PHX1 = x[6]
    PHX2 = x[7]
    PEX1 = x[8]
    PEX2 = x[9]
    PEX3 = x[10]
    PEX4 = x[11]
    PVX1 = x[12]
    PVX2 = x[13]
    F_z0 = x[-1]
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
        
    # Solve for some stuff before hand
    mu_x = (PDX1 + PDX2 * df_z) * lambda_mu_x
    BCD_x = F_z * (PKX1 + PKX2 * df_z) * np.exp(PKX3 * df_z)
        
    # Fit parameters to B C D E P_hy and P_vy
    D_x = (mu_x * F_z),
    C_x = PCX1,
    B_x = BCD_x / (mu_x * F_z * PCX1),
    S_hx = (PHX1 + PHX2 * df_z),
    E_x = ((PEX1 + PEX2 * df_z + PEX3 * df_z ** 2) * (1 - PEX4 * np.sign(s + S_hx))),
    S_vx = (F_z * (PVX1 + PVX2 * df_z))
    
    # FORGE THE MAGIC FORMULA
    Y = D_x * np.sin(C_x * np.arctan(B_x * (s + S_hx) - E_x * (B_x * (s + S_hx) - np.arctan(B_x * (s + S_hx))))) + S_vx
    return Y



#-------------------------------------------------------------------------------------------------
#                       LONGITUDINAL WITH COMBINED LOADING CORRECTION
#-------------------------------------------------------------------------------------------------



def first_pass_GX(data, x, long_params):
    # Read in the data
    F_z = -data["FZ"]
    F_x = data["FX"]
    s = data["SL"]
    alpha = np.tan(data["SA"] * np.pi / 180).mean()
    
    # Read in the parameters
    B_gx = x[0]
    C_gx = x[1]
    E_gx = x[2]
    S_hgx = x[3]
    
    # FORGE THE MAGIC FORMULA (for G_x)
    G_x0 = np.cos(C_gx * np.arctan(B_gx * S_hgx - E_gx * (B_gx * S_hgx - np.arctan(B_gx * S_hgx))))
    G_x = np.cos(C_gx * np.arctan(B_gx * (alpha + S_hgx) - E_gx * (B_gx * (alpha + S_hgx) - np.arctan(B_gx * (alpha + S_hgx))))) / G_x0
    
    # UTILIZE THE MAGIC FORMULA (for longitudinal force)
    Y = tm_long(F_z, s, 1, long_params) * G_x
    
    if (G_x.any() > 1):
        return (Y - F_x).squeeze() ** 10
    else: 
        return (Y - F_x).squeeze()

def second_pass_GX(data, F_z0, lambda_mu_x, BCES_params, x):    
    # Read in the parameters
    B_gx = BCES_params[:,0]
    C_gx = BCES_params[:,1]
    E_gx = BCES_params[:,2]
    S_hgx = BCES_params[:,3]
    
    # Read in the other parameters
    RBX1 = x[0]
    RBX2 = x[1]
    RBX3 = x[2]
    RCX1 = x[3]
    REX1 = x[4]
    REX2 = x[5]
    RHX1 = x[6]
    
    # Read in the data
    F_z = -data[0]["FZ"]
    s = data[0]["SL"]
    gamma = np.sin(data[0]["IA"] * np.pi / 180).mean()
    
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
    
    # Fit parameters to the other shit
    shit = (B_gx[0] - ((RBX1 + RBX3 * gamma ** 2) * np.cos(np.arctan(RBX2 * s))),
            C_gx[0] - RCX1,
            E_gx[0] - (REX1 + REX2 * df_z),
            S_hgx[0] - RHX1
            )
    
    residuals = np.vstack(shit)
    
    for i in range(1,len(data)):
        # Read in the data
        F_z = -data[i]["FZ"]
        s = data[i]["SL"]
        gamma = np.sin(data[i]["IA"] * np.pi / 180).mean()
        
        # Load sensitivity factor
        df_z = (F_z - F_z0) / F_z0
        
        # Fit parameters to the other shit
        shit = (residuals,
                B_gx[i] - ((RBX1 + RBX3 * gamma ** 2) * np.cos(np.arctan(RBX2 * s))),
                C_gx[i] - RCX1,
                E_gx[i] - (REX1 + REX2 * df_z),
                S_hgx[i] - RHX1
                )
        
        residuals = np.vstack(shit)
    
    return residuals.squeeze()
    
# # This is the data for Hoosier 20.5X7.0-13, R20; Rim_Width=7.0
# cases = []

# # Split up the data for the continuous sweeps
# split9_51 = sort(scipy.io.loadmat('data/straight_SI/B2356raw51.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_51, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])

# split9_52 = sort(scipy.io.loadmat('data/straight_SI/B2356raw52.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_52, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])
        
# BCES_params = np.zeros((len(cases),4))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0]
#     fit_func = lambda x:(first_pass_GX(cases[i],x,long_205X70_R20_70))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075],
#                                    [30, 2, 1, 0.075]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [10, 8, 0, BCES_params[:,1].mean(), 0, 0, BCES_params[:,3].mean()]

# fit_func = lambda x:(second_pass_GX(cases, long_205X70_R20_70[-1], 1, BCES_params, x))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# GX_205X70_R20_70 = result.x



# # This is the data for Hoosier 20.5X7.0-13, R20; Rim_Width=8.0
# cases = []

# # Split up the data for the continuous sweeps
# split9_54 = sort(scipy.io.loadmat('data/straight_SI/B2356raw54.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_54, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])
        
# split9_55 = sort(scipy.io.loadmat('data/straight_SI/B2356raw55.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_55, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])

# BCES_params = np.zeros((len(cases),4))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0]
#     fit_func = lambda x:(first_pass_GX(cases[i],x,long_205X70_R20_80))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075],
#                                    [30, 2, 1, 0.075]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [10, 8, 0, BCES_params[:,1].mean(), 0, 0, BCES_params[:,3].mean()]

# fit_func = lambda x:(second_pass_GX(cases, long_205X70_R20_80[-1], 1, BCES_params, x))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# GX_205X70_R20_80 = result.x



# # This is the data for Hoosier 18.0X6.0-10, R20; Rim_Width=6.0
# F_z0 = np.abs(np.vstack((scipy.io.loadmat('data/straight_SI/B2356raw69.mat')["FZ"],
#                          scipy.io.loadmat('data/straight_SI/B2356raw70.mat')["FZ"]))).mean()

# cases = []

# # Split up the data for the continuous sweeps
# split9_69 = sort(scipy.io.loadmat('data/straight_SI/B2356raw69.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_69, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])
        
# split9_70 = sort(scipy.io.loadmat('data/straight_SI/B2356raw70.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_70, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])
        
# BCES_params = np.zeros((len(cases),4))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0]
#     fit_func = lambda x:(first_pass_GX(cases[i],x,long_180X60_R20_60))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075],
#                                    [30, 2, 1, 0.075]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [10, 8, 0, BCES_params[:,1].mean(), 0, 0, BCES_params[:,3].mean()]

# fit_func = lambda x:(second_pass_GX(cases, long_180X60_R20_60[-1], 1, BCES_params, x))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# GX_180X60_R20_60 = np.hstack((result.x, F_z0))



# # This is the data for Hoosier 18.0X6.0-10, R20; Rim_Width=7.0
# cases = []

# # Split up the data for the continuous sweeps
# split9_72 = sort(scipy.io.loadmat('data/straight_SI/B2356raw72.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_72, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])

# split9 = sort(scipy.io.loadmat('data/straight_SI/B2356raw73.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.5):
#         cases.append(split[i])
        
# BCES_params = np.zeros((len(cases),4))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0]
#     fit_func = lambda x:(first_pass_GX(cases[i],x,long_180X60_R20_70))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075],
#                                    [30, 2, 1, 0.075]),
#                            # x_scale="jac",
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [10, 8, 0, BCES_params[:,1].mean(), 0, 0, BCES_params[:,3].mean()]

# fit_func = lambda x:(second_pass_GX(cases, long_180X60_R20_70[-1], 1, BCES_params, x))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+8), verbose=1)

# GX_180X60_R20_70 = np.hstack((result.x, F_z0))
   


#-------------------------------------------------------------------------------------------------
#                      DEFINE THE G CORRECTIONS FOR LATERAL MFS
#-------------------------------------------------------------------------------------------------



def first_pass_GY(data, x, lat_params):
    # Read in the data
    F_z = -data["FZ"]
    F_y = -data["FY"]
    s = data["SL"]
    gamma = np.sin(data["SA"] * np.pi / 180).mean()
    alpha = np.tan(data["SA"] * np.pi / 180)
    
    # Read in the parameters
    B_gy = x[0]
    C_gy = x[1]
    E_gy = x[2]
    S_hgy = x[3]
    S_vgy = x[4]
    
    # FORGE THE MAGIC FORMULA (for G_x)
    G_x0 = np.cos(C_gy * np.arctan(B_gy * S_hgy - E_gy * (B_gy * S_hgy - np.arctan(B_gy * S_hgy))))
    G_x = np.cos(C_gy * np.arctan(B_gy * (s + S_hgy) - E_gy * (B_gy * (s + S_hgy) - np.arctan(B_gy * (s + S_hgy))))) / G_x0
    
    # UTILIZE THE MAGIC FORMULA (for longitudinal force)
    Y, _, _, _, _, _ = tm_lat(F_z, alpha, gamma, 1, lat_params) * G_x
    
    if (G_x.any() > 1):
        return (Y - F_y + S_vgy).squeeze() ** 10
    else: 
        return (Y - F_y + S_vgy).squeeze()

def second_pass_GY(data, F_z0, lambda_mu_x, BCES_params, x, lat_params):    
    # Read in the parameters
    B_gy = BCES_params[:,0]
    C_gy = BCES_params[:,1]
    E_gy = BCES_params[:,2]
    S_hgy = BCES_params[:,3]
    S_vgy = BCES_params[:,4]
    
    # Read in the other parameters
    RBY1 = x[0]
    RBY2 = x[1]
    RBY3 = x[2]
    RBY4 = x[3]
    RCY1 = x[4]
    REY1 = x[5]
    REY2 = x[6]
    RHY1 = x[7]
    RHY2 = x[8]
    RVY1 = x[9] 
    RVY2 = x[10]
    RVY3 = x[11]
    RVY4 = x[12]
    RVY5 = x[13]
    RVY6 = x[14]
    
    # Read in the data
    F_z = -data[0]["FZ"]
    s = data[0]["SL"]
    gamma = np.sin(data[0]["IA"] * np.pi / 180).mean()
    alpha = np.tan(data[0]["SA"] * np.pi / 180)
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
    
    # Do some shit with the magic formula
    Y, _, _, _, _, mu_y = tm_lat(F_z, alpha, gamma, 1, lat_params)
    D_vgy = mu_y * F_z * (RVY1 + RVY2 * df_z + RVY3 * gamma) * np.cos(np.arctan(RVY4 * alpha))
    
    # Fit parameters to the other shit
    shit = (B_gy[0] - ((RBY1 + RBY4 * gamma ** 2) * np.cos(np.arctan(RBY2 * (alpha - RBY3)))),
            C_gy[0] - RCY1,
            E_gy[0] - (REY1 + REY2 * df_z),
            S_hgy[0] - (RHY1 + RHY2 * df_z),
            S_vgy[0] - (D_vgy * np.sin(RVY5 * np.arctan(RVY6 * s))) 
            )
    
    residuals = np.vstack(shit)
    
    for i in range(1,len(data)):
        # Read in the data
        F_z = -data[i]["FZ"]
        s = data[i]["SL"]
        gamma = np.sin(data[i]["IA"] * np.pi / 180).mean()
        alpha = np.tan(data[i]["SA"] * np.pi / 180).mean()
        
        # Load sensitivity factor
        df_z = (F_z - F_z0) / F_z0
        
        # Do some shit with the magic formula
        Y, _, _, _, _, mu_y = tm_lat(F_z, alpha, gamma, 1, lat_params)
        D_vgy = mu_y * F_z * (RVY1 + RVY2 * df_z + RVY3 * gamma) * np.cos(np.arctan(RVY4 * alpha))
        
        # Fit parameters to the other shit
        shit = (residuals,
                B_gy[i] - ((RBY1 + RBY4 * gamma ** 2) * np.cos(np.arctan(RBY2 * (alpha - RBY3)))),
                C_gy[i] - RCY1,
                E_gy[i] - (REY1 + REY2 * df_z),
                S_hgy[i] - (RHY1 + RHY2 * df_z),
                S_vgy[i] - (D_vgy * np.sin(RVY5 * np.arctan(RVY6 * s))) 
                )
        
        residuals = np.vstack(shit)
    
    return residuals.squeeze()
    
# # This is the data for Hoosier 20.5X7.0-13, R20; Rim_Width=7.0
# cases = []

# # Split up the data for the continuous sweeps
# split9_51 = sort(scipy.io.loadmat('data/straight_SI/B2356raw51.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_51, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])

# split9_52 = sort(scipy.io.loadmat('data/straight_SI/B2356raw52.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_52, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])
        
# BCES_params = np.zeros((len(cases),5))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0, 0]
#     fit_func = lambda x:(first_pass_GY(cases[i],x,lat_205X70_R20_70))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075, -500],
#                                    [30, 2, 1, 0.075, 500]),
#                            # x_scale="jac",4
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [7, 2.5, 0, 0, BCES_params[:,1].mean(), 0, 0, 0.02, 0, 0, 0, -0.2, 14, 1.9, 10]

# fit_func = lambda x:(second_pass_GY(cases, lat_205X70_R20_70[-1], 1, BCES_params, x, lat_205X70_R20_70))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+2), verbose=1)

# GY_205X70_R20_70 = result.x



# # This is the data for Hoosier 20.5X7.0-13, R20; Rim_Width=8.0
# cases = []

# # Split up the data for the continuous sweeps
# split9_54 = sort(scipy.io.loadmat('data/straight_SI/B2356raw54.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_54, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])

# split9_55 = sort(scipy.io.loadmat('data/straight_SI/B2356raw55.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_55, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])
        
# BCES_params = np.zeros((len(cases),5))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0, 0]
#     fit_func = lambda x:(first_pass_GY(cases[i],x,lat_205X70_R20_80))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075, -500],
#                                    [30, 2, 1, 0.075, 500]),
#                            # x_scale="jac",4
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [7, 2.5, 0, 0, BCES_params[:,1].mean(), 0, 0, 0.02, 0, 0, 0, -0.2, 14, 1.9, 10]

# fit_func = lambda x:(second_pass_GY(cases, lat_205X70_R20_80[-1], 1, BCES_params, x, lat_205X70_R20_80))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+2), verbose=1)

# GY_205X70_R20_80 = result.x



# # This is the data for Hoosier 18.0X6.0-10, R20; Rim_Width=6.0
# cases = []

# # Split up the data for the continuous sweeps
# split9_69 = sort(scipy.io.loadmat('data/straight_SI/B2356raw69.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_69, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])

# split9_70 = sort(scipy.io.loadmat('data/straight_SI/B2356raw70.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_70, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])
        
# BCES_params = np.zeros((len(cases),5))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0, 0]
#     fit_func = lambda x:(first_pass_GY(cases[i],x,lat_180X60_R20_60))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075, -500],
#                                    [30, 2, 1, 0.075, 500]),
#                            # x_scale="jac",4
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [7, 2.5, 0, 0, BCES_params[:,1].mean(), 0, 0, 0.02, 0, 0, 0, -0.2, 14, 1.9, 10]

# fit_func = lambda x:(second_pass_GY(cases, lat_180X60_R20_60[-1], 1, BCES_params, x, lat_180X60_R20_60))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+2), verbose=1)

# GY_180X60_R20_60 = result.x



# # This is the data for Hoosier 18.0X6.0-10, R20; Rim_Width=7.0
# cases = []

# # Split up the data for the continuous sweeps
# split9_72 = sort(scipy.io.loadmat('data/straight_SI/B2356raw72.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_72, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])

# split9_73 = sort(scipy.io.loadmat('data/straight_SI/B2356raw73.mat'), load_key="FZ", window=100, threshold_factor=10)
# split = bound(split9_73, slip_key="SL", threshold_factor=1)
# for i in range(len(split)):
#     if (split[i]["ET"].max()-split[i]["ET"].min()<15) & (split[i]["ET"].max()-split[i]["ET"].min()>5) & (np.abs(split[i]["SA"]).mean() > 0.8):
#         cases.append(split[i])
        
# BCES_params = np.zeros((len(cases),5))

# for i in range(len(cases)):
#     if i == 0:
#         x0_BCES = [10, 1.65, 0.75, 0, 0]
#     else:
#         x0_BCES = [10, BCES_params[i-1,1], 0.75, 0, 0]
#     fit_func = lambda x:(first_pass_GY(cases[i],x,lat_180X60_R20_70))
#     result = least_squares(fit_func, x0_BCES, jac='3-point', method='trf',
#                            bounds=([0, 1, 0, -0.075, -500],
#                                    [30, 2, 1, 0.075, 500]),
#                            # x_scale="jac",4
#                            # diff_step=1e-4,
#                            ftol=None, xtol=2.3e-16, gtol=2.3e-16,
#                            max_nfev=int(1e+8), verbose=1)
    
#     BCES_params[i] = result.x

# x0_R = [7, 2.5, 0, 0, BCES_params[:,1].mean(), 0, 0, 0.02, 0, 0, 0, -0.2, 14, 1.9, 10]

# fit_func = lambda x:(second_pass_GY(cases, lat_180X60_R20_70[-1], 1, BCES_params, x, lat_180X60_R20_70))
# result = least_squares(fit_func, x0_R, jac='3-point', method='lm',
#                        # x_scale="jac",
#                        # diff_step=1e-4,
#                        ftol=2.3e-16, xtol=2.3e-16, gtol=2.3e-16,
#                        max_nfev=int(1e+2), verbose=1)

# GY_180X60_R20_70 = result.x



#-------------------------------------------------------------------------------------------------
#                      DEFINE THE G CORRECTIONS FOR LONGITUDINAL AND LATERAL MFS
#-------------------------------------------------------------------------------------------------



def GX(F_z, F_z0, s, alpha, gamma, x):    
    # Read in the other parameters
    RBX1 = x[0]
    RBX2 = x[1]
    RBX3 = x[2]
    RCX1 = x[3]
    REX1 = x[4]
    REX2 = x[5]
    RHX1 = x[6]
    
    # Read in the data
    
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
    
    # Fit parameters to the other shit
    B_gx = ((RBX1 + RBX3 * gamma ** 2) * np.cos(np.arctan(RBX2 * s)))
    C_gx = RCX1
    E_gx = (REX1 + REX2 * df_z)
    S_hgx = RHX1
    
    # FORGE THE MAGIC FORMULA (for G_x)
    G_x0 = np.cos(C_gx * np.arctan(B_gx * S_hgx - E_gx * (B_gx * S_hgx - np.arctan(B_gx * S_hgx))))
    G_x = np.cos(C_gx * np.arctan(B_gx * (alpha + S_hgx) - E_gx * (B_gx * (alpha + S_hgx) - np.arctan(B_gx * (alpha + S_hgx))))) / G_x0
    
    
    return G_x

def GY(F_z, F_z0, s, alpha, gamma, x, lat_params):    
    # Read in the other parameters
    RBY1 = x[0]
    RBY2 = x[1]
    RBY3 = x[2]
    RBY4 = x[3]
    RCY1 = x[4]
    REY1 = x[5]
    REY2 = x[6]
    RHY1 = x[7]
    RHY2 = x[8]
    RVY1 = x[9] 
    RVY2 = x[10]
    RVY3 = x[11]
    RVY4 = x[12]
    RVY5 = x[13]
    RVY6 = x[14]
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
    
    # Do some shit with the magic formula
    Y, _, _, _, _, mu_y = tm_lat(F_z, alpha, gamma, 1, lat_params)
    D_vgy = mu_y * F_z * (RVY1 + RVY2 * df_z + RVY3 * gamma) * np.cos(np.arctan(RVY4 * alpha))
    
    # Fit parameters to the other shit
    B_gy = (RBY1 + RBY4 * gamma ** 2) * np.cos(np.arctan(RBY2 * (alpha - RBY3)))
    C_gy = RCY1
    E_gy = REY1 + REY2 * df_z
    S_hgy = RHY1 + RHY2 * df_z
    S_vgy = D_vgy * np.sin(RVY5 * np.arctan(RVY6 * s))
    
    # FORGE THE MAGIC FORMULA (for G_x)
    G_x0 = np.cos(C_gy * np.arctan(B_gy * S_hgy - E_gy * (B_gy * S_hgy - np.arctan(B_gy * S_hgy))))
    G_x = np.cos(C_gy * np.arctan(B_gy * (s + S_hgy) - E_gy * (B_gy * (s + S_hgy) - np.arctan(B_gy * (s + S_hgy))))) / G_x0

    return G_x, S_vgy
    


#-------------------------------------------------------------------------------------------------
#                    DETERMINE THE OVERTURNING MOMENT   
#-------------------------------------------------------------------------------------------------



def fit_MX(data, x, lat_params, gy_params):
    # Read in the fitted shit
    QSX1 = x[0]
    QSX2 = x[1]
    QSX3 = x[2]
    
    # Get the unloaded radius and nominal load
    F_z0 = lat_params[-1]
    R_0 = lat_params[-2] * 0.5 * 0.0254 # [m]
    
    # Read in the data
    M_x = data[0]["MX"]
    F_z = -data[0]["FZ"]
    gamma = np.sin(data[0]["IA"] * np.pi / 180)
    alpha = np.tan(data[0]["SA"] * np.pi / 180)
    s = data[0]["SL"]
    
    # Compute the corrected longitudinal force
    Y = tm_lat(F_z, alpha, gamma, 1, lat_params)
    G_y = GY(F_z, F_z0, s, alpha, gamma, gy_params, lat_params)
    F_y = Y * G_y
    
    # COMPUTE THE MAGIC OVERTURING MOMENT
    residuals = (F_z * R_0 * (QSX1 - QSX2 * gamma + QSX3 * F_y / F_z0) - M_x).squeeze()
    
    for i in range(1,len(data)):
        # Read in the data
        M_x = data[i]["MX"]
        F_z = -data[i]["FZ"]
        gamma = np.sin(data[i]["IA"] * np.pi / 180)
        alpha = np.tan(data[i]["SA"] * np.pi / 180)
        s = data[i]["SL"]
        
        # Compute the corrected longitudinal force
        Y = tm_lat(F_z, alpha, gamma, 1, lat_params)
        G_y = GY(F_z, F_z0, s, alpha, gamma, gy_params, lat_params)
        F_y = Y * G_y
        
        # COMPUTE THE MAGIC OVERTURING MOMENT
        residuals = np.vstack((residuals, (F_z * R_0 * (QSX1 - QSX2 * gamma + QSX3 * F_y / F_z0) - M_x).squeeze()))
        
    
    return residuals.squeeze()



#-------------------------------------------------------------------------------------------------
#                    DETERMINE THE ROLLING RESISTANCE MOMENT   
#-------------------------------------------------------------------------------------------------



def fit_MY(data, x, long_params, gx_params):
    # Read in the fitted shit
    QSY1 = x[0]
    QSY2 = x[1]
    
    # Get the unloaded radius and nominal load
    F_z0 = long_params[-1]
    R_0 = long_params[-2] * 0.5 * 0.0254 # [m]
    
    # Compute reference velocity using gravity
    v_0 = np.sqrt(9.81 * R_0)
    
    # Read in the data
    M_y = data[0]["MY"]
    F_z = -data[0]["FZ"]
    gamma = np.sin(data[0]["IA"] * np.pi / 180)
    alpha = np.tan(data[0]["SA"] * np.pi / 180)
    s = data[0]["SL"]
    v_r = data[0]["V"] * 0.277778
    
    # Compute the corrected longitudinal force
    Y = tm_long(F_z, alpha, gamma, 1, long_params)
    G_x = GX(F_z, F_z0, s, alpha, gamma, gx_params)
    F_x = Y * G_x
    
    # COMPUTE THE MAGIC OVERTURING MOMENT
    residuals = (-F_z * R_0 * (QSY1 * np.arctan(v_r / v_0) + QSY2 * F_x / F_z0) - M_y).squeeze()
    
    for i in range(1,len(data)):
        # Read in the data
        M_y = data[i]["MY"]
        F_z = -data[i]["FZ"]
        gamma = np.sin(data[i]["IA"] * np.pi / 180)
        alpha = np.tan(data[i]["SA"] * np.pi / 180)
        s = data[i]["SL"]
        v_r = data[i]["V"] * 0.277778
        
        # Compute the corrected longitudinal force
        Y = tm_long(F_z, alpha, gamma, 1, long_params)
        G_x = GX(F_z, F_z0, s, alpha, gamma, gx_params)
        F_x = Y * G_x
        
        # COMPUTE THE MAGIC OVERTURING MOMENT
        residuals = np.vstack((residuals, (-F_z * R_0 * (QSY1 * np.arctan(v_r / v_0) + QSY2 * F_x / F_z0) - M_y).squeeze()))
        
    
    return residuals.squeeze()












# thing = cases[100]
# F_z = thing["FZ"]
# alpha = thing["SA"]
# gamma = thing["IA"]
# F_y = thing["FY"]
# Y = tm(F_z, F_z0, alpha, gamma, 1, lateral_parameters)
# plt.scatter(alpha, F_y, color="black", label="data")
# plt.scatter(alpha,Y,color="red",label="MF",lw=2)


# alpha = np.linspace(-10,10,5000)
# gamma=10
# for i in range(int(F_z0), 2000, 300):
#     Y = tm(i, F_z0, alpha, gamma, 1, lateral_parameters)
#     plt.plot(alpha,Y,color="red",label=f"F$_z$={i} [N]",lw=2)
# plt.legend(fontsize=16)
# plt.xlabel("Slip Angle [°]",fontsize=20)
# plt.ylabel("F$_y$ [N]",fontsize=20)
# plt.tick_params(axis="both",labelsize=14)
# plt.grid(True, linestyle=':', alpha=0.7)