# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.constants as const

# # Import your physics engine
# from core_gemini import solve_first_eigenfunction, compute_density_profile, B_z

# # ==========================================
# # 1. WHAM++ ABSOLUTE PARAMETERS
# # ==========================================
# B_0 = 2.5              # Central cell magnetic field [Tesla]
# R_M = 10.0             # Mirror ratio
# L = 2.5                # Half-length of the device [meters]
# a = 0.3                # Midplane plasma radius [meters]
# beta_target = 0.5      # Target normalized plasma pressure
# T_i_eff_keV = 75.4     # Effective ion temperature [keV] 
# T_e_keV = 16.3         # Electron temperature [keV] 

# # Calculate absolute average density <n> 
# T_total_Joules = (T_i_eff_keV + T_e_keV) * 1e3 * const.e
# n_target = beta_target * (B_0**2) / (2 * const.mu_0 * T_total_Joules)

# # ==========================================
# # 2. BOSCH-HALE D-T FUSION REACTIVITY
# # ==========================================
# def get_sigma_v_DT(T_keV):
#     """Calculates D-T fusion reactivity <sigma v> in m^3/s using Bosch-Hale (1992)."""
#     # Bosch-Hale parameters for D-T
#     c1, c2, c3 = 1.17302e-9, 1.51361e-2, 7.51886e-2
#     c4, c5, c6, c7 = 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5
#     BG = 34.3827
#     mc2 = 1124656.0 # Reduced mass energy [keV]
    
#     T = T_keV
#     theta = T / (1.0 - (T * (c2 + T * (c4 + T * c6))) / (1.0 + T * (c3 + T * (c5 + T * c7))))
#     xi = (BG**2 / (4.0 * theta))**(1.0/3.0)
    
#     # Reactivity in cm^3/s, then convert to m^3/s
#     sigma_v_cm3_s = c1 * theta * np.sqrt(xi / (mc2 * T**3)) * np.exp(-3.0 * xi)
#     return sigma_v_cm3_s * 1e-6

# # Get the constant reactivity for our effective temperature
# sigma_v = get_sigma_v_DT(T_i_eff_keV)
# print(f"Calculated D-T Reactivity at {T_i_eff_keV} keV: {sigma_v:.2e} m^3/s")

# # ==========================================
# # 3. GENERATE SPATIAL PROFILES
# # ==========================================
# k_val, w_val = 1.3, 0.1

# print("Solving spatial density profile...")
# L_grid, I_1 = solve_first_eigenfunction(R_M=R_M, k=k_val, w=w_val)
# z_norm = np.linspace(0, 1, 200)

# # Get normalized relative density, then scale to physical absolute density
# #n_rel = compute_density_profile(z_norm, R_M, k_val, w_val, L_grid, I_1)
# Lambda_inj = np.sin(np.radians(45))**2  # This equals 0.5
# L_grid = np.linspace(1.0/R_M + 1e-5, 0.999, 300)
# spread = 0.05
# I_45deg = np.exp(-((L_grid - Lambda_inj)**2) / (2 * spread**2))
# I_45deg += np.where(L_grid < Lambda_inj, 0.2, 0.0) # Add a flat scattering tail down to the loss cone

# n_rel = compute_density_profile(z_norm, R_M, k_val, w_val, L_grid, I_45deg)
# n_abs_half = n_rel * n_target
# n_abs = np.concatenate((n_abs_half[::-1], n_abs_half[1:]))

# # Get B-field and physical z-axis
# B_half = B_z(z_norm, R_M, k_val, w_val) * B_0
# B_abs = np.concatenate((B_half[::-1], B_half[1:]))
# z_phys = np.concatenate((-z_norm[::-1] * L, z_norm[1:] * L))

# # ==========================================
# # 4. NEUTRON SOURCE & TOTAL YIELD
# # ==========================================
# # Local neutron source rate: S_n(z) = (n_D * n_T) * <sigma v> = (n^2 / 4) * <sigma v>
# S_n = (n_abs**2 / 4.0) * sigma_v

# # Integrate over the flux tube volume to get total neutrons/sec
# # dV = A(z) dz = (A_0 * B_0 / B(z)) dz
# A_0 = np.pi * a**2
# dV = (A_0 * B_0 / B_abs) * (z_phys[1] - z_phys[0])
# Total_Neutrons_per_sec = np.sum(S_n * dV)

# # Fusion Power (Assuming 17.6 MeV per D-T reaction)
# # Note: Egedal assumes 22.4 MeV including blanket breeding, but 17.6 is standard raw fusion
# Energy_per_reaction_Joules = 17.6e6 * const.e
# Total_Fusion_Power_MW = (Total_Neutrons_per_sec * Energy_per_reaction_Joules) / 1e6

# # ==========================================
# # 5. PLOTTING
# # ==========================================
# fig, ax = plt.subplots(figsize=(8, 5))

# ax.plot(z_phys, S_n, color='darkorange', linewidth=2.5)
# ax.fill_between(z_phys, S_n, color='darkorange', alpha=0.3)

# ax.set_xlim(-L, L)
# ax.set_ylim(0, np.max(S_n) * 1.1)
# ax.set_xlabel('Physical Axial Position, z [meters]', fontsize=12)
# ax.set_ylabel('Neutron Source Rate, $S_n(z)$ [n / m$^3$ / s]', fontsize=12)
# ax.set_title('WHAM++ Axial Neutron Source Profile', fontsize=14)
# ax.grid(True, alpha=0.3)

# # Add a text box with the integrated reactor metrics
# textstr = '\n'.join((
#     f'Peak Rate: {np.max(S_n):.2e} $n/m^3/s$',
#     f'Total Yield: {Total_Neutrons_per_sec:.2e} $n/s$',
#     f'Fusion Power: {Total_Fusion_Power_MW:.2f} MW'
# ))
# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
#         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# plt.tight_layout()
# plt.savefig('neutron_source_experiment.png', dpi=300, bbox_inches='tight')
# plt.close()

# print("Neutron source experiment plot saved as 'neutron_source_experiment.png'")

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# Import your physics engine
from core_gemini import solve_first_eigenfunction, compute_density_profile, B_z

# ==========================================
# 1. WHAM++ ABSOLUTE PARAMETERS
# ==========================================
B_0 = 2.5              # Central cell magnetic field [Tesla]
R_M = 10.0             # Mirror ratio
L = 2.5                # Half-length of the device [meters]
a = 0.3                # Midplane plasma radius [meters]
beta_target = 0.5      # Target normalized plasma pressure
T_i_eff_keV = 75.4     # Effective ion temperature [keV] 
T_e_keV = 16.3         # Electron temperature [keV] 

# Calculate absolute average density <n> 
T_total_Joules = (T_i_eff_keV + T_e_keV) * 1e3 * const.e
n_target = beta_target * (B_0**2) / (2 * const.mu_0 * T_total_Joules)

# ==========================================
# 2. BOSCH-HALE D-T FUSION REACTIVITY
# ==========================================
def get_sigma_v_DT(T_keV):
    """Calculates D-T fusion reactivity <sigma v> in m^3/s using Bosch-Hale (1992)."""
    c1, c2, c3 = 1.17302e-9, 1.51361e-2, 7.51886e-2
    c4, c5, c6, c7 = 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5
    BG = 34.3827
    mc2 = 1124656.0 # Reduced mass energy [keV]
    
    T = T_keV
    theta = T / (1.0 - (T * (c2 + T * (c4 + T * c6))) / (1.0 + T * (c3 + T * (c5 + T * c7))))
    xi = (BG**2 / (4.0 * theta))**(1.0/3.0)
    
    sigma_v_cm3_s = c1 * theta * np.sqrt(xi / (mc2 * T**3)) * np.exp(-3.0 * xi)
    return sigma_v_cm3_s * 1e-6

sigma_v = get_sigma_v_DT(T_i_eff_keV)

# ==========================================
# 3. GENERATE SPATIAL PROFILES (WITH DCLC WARM IONS)
# ==========================================
k_val, w_val = 1.3, 0.1
z_norm = np.linspace(0, 1, 200)

# 1. Base 45-degree fast ion distribution
Lambda_inj = np.sin(np.radians(45))**2  
L_grid = np.linspace(1.0/R_M + 1e-5, 0.999, 300)
spread = 0.15
I_45deg = np.exp(-((L_grid - Lambda_inj)**2) / (2 * spread**2))
I_45deg += np.where(L_grid < Lambda_inj, 0.2, 0.0) 

# Get the relative fast-ion density
n_fast_rel = compute_density_profile(z_norm, R_M, k_val, w_val, L_grid, I_45deg)

# 2. FIX 1: DCLC Warm Ion Trap Fill
peak_fast_density = np.max(n_fast_rel)
midplane_fast_density = n_fast_rel[0]
hole_depth = peak_fast_density - midplane_fast_density

# Create a warm ion population to precisely fill the midplane valley
warm_spread = 1.2
n_warm_rel = hole_depth * np.exp(-(z_norm**2) / (2 * warm_spread**2))

# The total DCLC-stabilized relative density
n_total_rel = n_fast_rel + n_warm_rel

# 3. Scale to absolute physical density
n_abs_half = n_total_rel * n_target
n_abs = np.concatenate((n_abs_half[::-1], n_abs_half[1:]))

B_half = B_z(z_norm, R_M, k_val, w_val) * B_0
B_abs = np.concatenate((B_half[::-1], B_half[1:]))
z_phys = np.concatenate((-z_norm[::-1] * L, z_norm[1:] * L))

# ==========================================
# 4. NEUTRON SOURCE & WALL FLUX (SPINDLE GEOMETRY)
# ==========================================
# Local volumetric source rate
S_n = (n_abs**2 / 4.0) * sigma_v

# FIX 2: The Spindle Geometry - Plasma radius shrinks as B increases
r_plasma = a * np.sqrt(B_0 / B_abs)

# Neutrons produced per unit length (Neutrons / m / s)
Neutrons_per_meter = S_n * (np.pi * r_plasma**2)

# Neutron Wall Flux (Neutrons / m^2 / s) - Assuming a 40cm cylindrical blanket wall
r_wall = 0.40 
Wall_Area_per_meter = 2 * np.pi * r_wall
Neutron_Wall_Flux = Neutrons_per_meter / Wall_Area_per_meter

# Total Yield 
dV = (np.pi * r_plasma**2) * (z_phys[1] - z_phys[0])
Total_Neutrons_per_sec = np.sum(S_n * dV)
Total_Fusion_Power_MW = (Total_Neutrons_per_sec * 17.6e6 * const.e) / 1e6

# ==========================================
# 5. PLOTTING
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the FLUX hitting the wall, not the volumetric source
ax.plot(z_phys, Neutron_Wall_Flux, color='purple', linewidth=2.5)
ax.fill_between(z_phys, Neutron_Wall_Flux, color='purple', alpha=0.3)

ax.set_xlim(-L, L)
ax.set_ylim(0, np.max(Neutron_Wall_Flux) * 1.1)
ax.set_xlabel('Physical Axial Position, z [meters]', fontsize=12)
ax.set_ylabel('Neutron Wall Flux [n / m$^2$ / s]', fontsize=12)
ax.set_title('WHAM++ Realistic Neutron Wall Flux (45° Stabilized)', fontsize=14)
ax.grid(True, alpha=0.3)

textstr = '\n'.join((
    f'Peak Wall Flux: {np.max(Neutron_Wall_Flux):.2e} $n/m^2/s$',
    f'Total Yield: {Total_Neutrons_per_sec:.2e} $n/s$',
    f'Fusion Power: {Total_Fusion_Power_MW:.2f} MW'
))
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('neutron_source_experiment_lol.png', dpi=300, bbox_inches='tight')
plt.close()

print("Corrected neutron wall flux plot saved as 'neutron_source_experiment.png'")
