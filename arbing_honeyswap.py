# Based on https://hackmd.io/@markusbkoch/SyfNfQURw
from math import sqrt

def swap(y_1, Y_1, Y_2, sigma):
    y_2 = Y_2 * y_1 * (1 - sigma)
    y_2 /= Y_1 + 2 * y_1 * (1 - sigma)
    return y_2


def arbitrage(x_1, P_x, P_z, rho_1, sigma_x, sigma_z, c, K_x, K_z, g_x, g_z):
    # Swap for token '2' on AMM x
    X_1 = sqrt(P_x * K_x)
    X_2 = sqrt(K_x / P_x)
    x_2 = swap(x_1, X_1, X_2, sigma_x)
    z_2 = x_2

    # Swap for token '1' on AMM y
    Z_1 = sqrt(P_z * K_z)
    Z_2 = sqrt(K_z / P_z)
    z_1 = swap(z_2, Z_2, Z_1, sigma_z)

    # Raw profit
    R = z_1 - x_1

    # net profit
    E = rho_1 * R - c * (g_x + g_z)
    return E

# Parameters for AMM 1
P_x = 1
K_x = 1e6
sigma_x = 0.03

# Parameters for AMM 2
P_z = 1
K_z = 1e6
sigma_z = 0.03

# ETH price for token 1
rho_1 = 1.0

# ETH paid per unit gas
c = 63e-9 # 63 Gwei

# Fee for swaping on AMM x and z
g_x = 20 # 20 * base_fee
g_z = 30 # 30 * base_fee

# Try to arbitrage with 1.0 of token '1'
x_1 = 1.0
args = (x_1, P_x, P_z, rho_1, sigma_x, sigma_z, c, K_x, K_z, g_x, g_z)
print(arbitrage(*args))