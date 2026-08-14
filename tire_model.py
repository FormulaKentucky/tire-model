import numpy as np
from matplotlib import pyplot as plt

# F_z = float(input("Enter vertical load on tire [N]: "))
# gamma = float(input("Enter inclination angle [°]: "))
# lambda_mu = float(input("Enter road roughness (0-1): "))

lat_160X75_R20_70 = [2.91209, 0.211056, 9.97357, 1.52932, 44.358, -0.645862, 118.447,
                     -1.13755, 40.4831, 0.00676744, 0.00301196, 0.00220077, 0.00190766,
                     -213.228, 386.147, -4407.4, -0.00102474, -0.0131883, 4.20393,
                     1.41215, 0.0, -740.577]

lat_160X75_R20_80 = [2.89436, 0.206051, 15.5546, 1.46299, 45.345, 208.122, 33.755,
                     318.556, 7044.28, 0.0044863, 0.00179675, 0.00466722, 0.000775571,
                     -127.882, 67.5115, -1970.26, -0.0282984, -0.0224862, 1.8986,
                     0.625785, 0.0, -733.268]

long_180X60_R20_60 = [2.80138, 0.186458, 1.68934, 75.9055, -6.92883e-06, 0.135873,
                      0.00556941, 0.00342396, 0.713805, 0.161747, 0.0334031, -0.00512696,
                      0.134194, 0.0431431, -707.14]

long_180X60_R20_70 = [2.5735, 0.145952, 1.66414, 70.5789, 0.016365, 0.134844, 0.00472402,
                      0.00312736, 2.48957, 2.02391, 0.485028, 0.000311031, 0.113012,
                      0.0362511, -700.511]

long_205X70_R20_70 = [2.14749, -0.0492872, 1.81658, 63.2464, -4.12794e-05, 0.000601728,
                      0.0134841, 0.00853666, 2.09856, 1.44784, 0.338257, -0.00525735,
                      -0.00525735, 0.0524493, -814.792]

long_205X70_R20_80 = [2.58207, 0.180327, 1.77441, 79.9478, 0.000671284, 0.166922,
                      0.00707393, 0.00515334, 1.42553, 0.767869, 0.158239, -0.0112999,
                      0.0890588, 0.00927758, -816.53]

params = [lat_160X75_R20_70, lat_160X75_R20_80, long_180X60_R20_60, long_180X60_R20_70,
          long_205X70_R20_70, long_205X70_R20_80]



def tm_lat(F_z, F_z0, alpha, gamma, lambda_mu_y, x):    
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
    PHY1 = x[9]
    PHY2 = x[10]
    PEY1 = x[11]
    PEY2 = x[12]
    PEY3 = x[13]
    PEY4 = x[14]
    PEY5 = x[15]
    PVY1 = x[16]
    PVY2 = x[17]
    PVY3 = x[18]
    PVY4 = x[19]
        
    # Initialize arrays
    alpha = np.tan(alpha * np.pi / 180)
    gamma = np.sin(gamma * np.pi / 180)
    
    # Load sensitivity factor
    df_z = (F_z - F_z0) / F_z0
        
    # Solve for some stuff before hand
    mu_y = (PDY1 + PDY2 * df_z) * lambda_mu_y / (1 + PDY3 * gamma ** 2) 
    BCD_y = PKY1 * F_z0 * np.sin(PKY4 * np.arctan(F_z / ((PKY2 + PKY5 * gamma ** 2) * F_z0)) / (1 + PKY3 * gamma ** 2))
    S_vy_gamma = F_z * (PVY3 + PVY4 * df_z) * gamma
        
    # Fit parameters to B C D E P_hy and P_vy
    D_y = (mu_y * F_z)
    C_y = PCY1
    B_y = BCD_y / (mu_y * F_z * PCY1)
    S_hy = (PHY1 + PHY2 * df_z)
    E_y = ((PEY1 + PEY2 * df_z) * (1 + PEY5 * gamma ** 2 - (PEY3 + PEY4 * gamma) - np.sign(alpha + (PHY1 + PHY2 * df_z))))
    S_vy = ((PVY1 + PVY2 * df_z) * F_z + S_vy_gamma)
    
    # FORGE THE MAGIC FORMULA
    Y = D_y * np.sin(C_y * np.arctan(B_y * (alpha + S_hy) - E_y * (B_y * (alpha + S_hy) - np.arctan(B_y * (alpha + S_hy))))) + S_vy
    return Y.squeeze()


def tm_long(F_z, F_z0, s, lambda_mu_x, x):        
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
    return Y.squeeze()


# alpha = np.linspace(-10,10,5000)
# Y = tm_lat(F_z, params[0][-1], alpha, gamma, lambda_mu, params[0][:-1])
# plt.plot(alpha,Y,color="red",label=f"F$_z$={F_z} [N]",lw=2)
# plt.legend(fontsize=16)
# plt.xlabel("Slip Angle [°]",fontsize=20)
# plt.ylabel("F$_y$ [N]",fontsize=20)
# plt.tick_params(axis="both",labelsize=14)
# plt.grid(True, linestyle=':', alpha=0.7)

s = np.linspace(-0.2,0.2,5000)
for i in range(700, 3000, 300):
    F_z = i
    gamma = 2
    Y = tm_long(F_z, params[2][-1], s, 1, params[2][:-1])
    plt.plot(s,Y,label=f"F$_z$={F_z} [N]",lw=2)
    plt.legend(fontsize=16)
    plt.xlabel("Slip Ratio [-]",fontsize=20)
    plt.ylabel("F$_x$ [N]",fontsize=20)
    plt.tick_params(axis="both",labelsize=14)
    plt.title("long_180X60_R20_60 Vertical Load Sweep", fontsize=28, fontweight="bold")
    plt.grid(True, linestyle=':', alpha=0.7)