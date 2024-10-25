import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
# Example with rapeseed oil as fluid and quartzite rocks as solids
## Reactor parameters
L = 1.8  # Reactor length (m)
D = 0.4  # Reactor diameter (m)
epsilon = 0.41  # Bed void fraction, usually between 0.2 and 0.4
Ds = 0.04 # Particle diameter (m)
rho_f = 850  # Fluid density (kg/m^3)
Cp_f = 2400  # Fluid heat capacity (J/kg/K)
mu_f = 0.002 # Fluid dynamic viscosity (kg/m/s)
rho_s = 2500  # Solid density (kg/m^3)
Cp_s = 830  # Solid heat capacity (J/kg/K)
rho_w = 7800  # Wall density (kg/m^3)
Cp_w = 500  # Wall heat capacity (J/kg/K)
t_w = 0.005  # Wall thickness (m)
Q_f = 0.019 # Fluid flow rate (kg/s)
lambda_f = 0.15 # Fluid thermal conductivity (W/m/K)
lambda_s = 5.69 # Solids thermal conductivity (W/m/K)

## Operating conditions
T_in = 273 + 160  # Inlet temperature (K)
T_s0 = 273 + 210  # Initial solid temperature (K)
T_w0 = 350  # Initial wall temperature (K)
T_amb = 298  # Ambient temperature (K)

## Calculated parameters
A = np.pi * D**2 / 4  # Cross-sectional area (m^2)
a_fs = 6 * (1 - epsilon) / Ds  # Specific surface area (m^2/m^3)
a_fw = 4 / D  # Specific surface area for fluid-wall heat transfer (m^2/m^3)
u = Q_f / (rho_f * epsilon * np.pi * D ** 2 / 4)  # Interstitial velocity (m/s)
u_sup = epsilon * u # Superficial fluid velocity (m/s)
Re = rho_f * u_sup * Ds / mu_f # Reynolds number
Pr = mu_f * Cp_f / lambda_f # Prandtl number
Nu = 2 + 1.8 * Re**0.5 * Pr**(1/3) # Nusselt number, correlation by Ranz (1952)
h_fs = Nu * lambda_f / Ds # Heat transfer coefficient (W/m^2/K)
Bi = h_fs * Ds / (6 * lambda_s) # Biot number, must be < 0.1 for solids thermal gradient to be negligible

# to be calculated
h_wa = 10  # Heat transfer coefficient between wall and ambient (W/m^2/K)
h_fw = 50  # Heat transfer coefficient between fluid and wall (W/m^2/K) 

## Discretization
Nz = 100 # Number of axial grid points
dz = L / (Nz - 1)  # Grid spacing
z = np.linspace(0, L, Nz)  # Axial coordinate

def schumann_model(y, t, u, T_in):
    T_f = y[:Nz]
    T_s = y[Nz:2*Nz]
    T_w = y[2*Nz:]
    
    dTf_dt = np.zeros(Nz)
    dTs_dt = np.zeros(Nz)
    dTw_dt = np.zeros(Nz)
    
    # Fluid phase energy balance
    dTf_dt[0] = u / dz * (T_in - T_f[0]) - h_fs * a_fs / (rho_f * Cp_f) * (T_f[0] - T_s[0]) - h_fw * a_fw / (rho_f * Cp_f) * (T_f[0] - T_w[0])
    dTf_dt[1:] = u / dz * (T_f[:-1] - T_f[1:]) - h_fs * a_fs / (rho_f * Cp_f) * (T_f[1:] - T_s[1:]) - h_fw * a_fw / (rho_f * Cp_f) * (T_f[1:] - T_w[1:])
    
    # Solid phase energy balance
    dTs_dt = h_fs * a_fs / (rho_s * Cp_s) * (T_f - T_s)
    
    # Wall energy balance (without axial conduction)
    dTw_dt = (h_fw * a_fw * (T_f - T_w) - h_wa * 4 / D * (T_w - T_amb)) / (rho_w * Cp_w * t_w)
    
    return np.concatenate((dTf_dt, dTs_dt, dTw_dt))

## Initial conditions
y0 = np.concatenate((np.ones(Nz) * T_in, np.ones(Nz) * T_s0, np.ones(Nz) * T_w0))

## Time span (start, stop, number of points)
t_span = np.linspace(0, 0.5*3600, 1000)

## Solve ODE system
solution = odeint(schumann_model, y0, t_span, args=(u, T_in))

## Extract results
T_f = solution[:, :Nz]
T_s = solution[:, Nz:2*Nz]
T_w = solution[:, 2*Nz:]

## Plot results
# =============================================================================
# plt.figure(figsize=(10, 6))
# plt.plot(z, T_f[-1, :] - 273, label='Fluid temperature')
# plt.plot(z, T_s[-1, :] - 273, label='Solid temperature')
# plt.plot(z, T_w[-1, :] - 273, label='Wall temperature')
# plt.xlabel('Axial position (m)')
# plt.ylabel('Temperature (°C)')
# plt.title('Schumann Model - Packed Bed Reactor')
# plt.legend()
# plt.grid(True)
# plt.show()
# =============================================================================
# Plot temperature evolution over time at different reactor positions
positions = [0, Nz//4, Nz//2, 3*Nz//4, -1]  # Start, 1/4, 1/2, 3/4, End
plt.figure(figsize=(12, 8))
for pos in positions:
    plt.plot(t_span, T_f[:, pos], label=f'Fluid at z={z[pos]:.2f}m')
    plt.plot(t_span, T_s[:, pos], '--', label=f'Solid at z={z[pos]:.2f}m')
    plt.plot(t_span, T_w[:, pos], ':', label=f'Wall at z={z[pos]:.2f}m')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Temperature Evolution Over Time at Different Reactor Positions')
plt.legend()
plt.grid(True)
plt.show()