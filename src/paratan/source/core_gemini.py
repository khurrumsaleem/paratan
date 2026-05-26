import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import root_scalar
import scipy.linalg as la
from scipy.interpolate import interp1d
import warnings

# Suppress harmless integration warnings near the singularities
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CORE PHYSICS FUNCTIONS
# ==========================================

def B_z(z, R_M, k, w):
    term1 = np.exp(- (np.abs(np.abs(z) - 1)**k) / (w**k))
    term2 = np.exp(- 1 / (w**k))
    return 1 + (R_M - 1) * ((term1 - term2) / (1 - term2))

def find_turning_point(Lambda, R_M, k, w):
    if Lambda <= 1.0 / R_M: return 1.0
    res = root_scalar(lambda z: B_z(z, R_M, k, w) - 1.0/Lambda, bracket=[0, 1])
    return res.root

def compute_tau_b(Lambda, R_M, k, w):
    if Lambda == 0: return 1.0
    z_tp = find_turning_point(Lambda, R_M, k, w)
    def integrand(z):
        val = 1.0 - Lambda * B_z(z, R_M, k, w)
        return 1.0 / np.sqrt(val) if val > 0 else 0.0
    res, _ = quad(integrand, 0, z_tp, limit=100)
    return res

def compute_avg_invB(Lambda, R_M, k, w):
    if Lambda <= 1.0 / R_M: return 1.0 / R_M 
    z_tp = find_turning_point(Lambda, R_M, k, w)
    def integrand_num(z):
        B = B_z(z, R_M, k, w)
        val = 1.0 - Lambda * B
        return (1.0 / B) / np.sqrt(val) if val > 0 else 0.0
    num, _ = quad(integrand_num, 0, z_tp, limit=100)
    den = compute_tau_b(Lambda, R_M, k, w) 
    return num / den

def solve_first_eigenfunction(R_M, k, w, N_points=100):
    Lambda_min = 1.0 / R_M
    L_grid = np.linspace(Lambda_min + 1e-5, 0.999, N_points)
    dL = L_grid[1] - L_grid[0]
    
    avg_invB = np.array([compute_avg_invB(L, R_M, k, w) for L in L_grid])
    C1 = 4 * avg_invB - 6 * L_grid
    C2 = 4 * L_grid * (avg_invB - L_grid)
    
    L_mat = np.zeros((N_points, N_points))
    for i in range(1, N_points - 1):
        L_mat[i, i-1] = C2[i]/(dL**2) - C1[i]/(2*dL)
        L_mat[i, i]   = -2*C2[i]/(dL**2)
        L_mat[i, i+1] = C2[i]/(dL**2) + C1[i]/(2*dL)
        
    L_mat[0, 0] = -2*C2[0]/(dL**2)
    L_mat[0, 1] = C2[0]/(dL**2) + C1[0]/(2*dL)
    L_mat[-1, -2] = 2 * C2[-1]/(dL**2)
    L_mat[-1, -1] = -2 * C2[-1]/(dL**2)
    
    eigenvalues, eigenvectors = la.eig(L_mat)
    real_evals = np.real(eigenvalues)
    valid_idx = np.where(real_evals < -1e-5)[0] 
    first_mode_idx = valid_idx[np.argmax(real_evals[valid_idx])] 
    
    I_1 = np.real(eigenvectors[:, first_mode_idx])
    I_1 = I_1 / I_1[-1] # Normalize
    
    L_grid_full = np.insert(L_grid, 0, Lambda_min)
    I_1_full = np.insert(I_1, 0, 0.0)
    return L_grid_full, I_1_full

# --- 5. Projecting back to Density Space (Updated) ---

def compute_density_profile(z_array, R_M, k, w, L_grid, I_1):
    """Computes the spatial density profile n(z)/<n>."""
    
    # FIX 1: Hold the midplane value for Lambda > 0.999 instead of dropping to 0
    I_1_interp = interp1d(L_grid, I_1, kind='linear', bounds_error=False, fill_value=(0.0, I_1[-1]))

    tau_b_grid = np.array([compute_tau_b(L, R_M, k, w) for L in L_grid])
    denominator = np.trapezoid(tau_b_grid * I_1, L_grid)
    
    L_full = np.linspace(0, 0.999, 300)
    tau_b_full = np.array([compute_tau_b(L, R_M, k, w) for L in L_full])
    avg_1_Lambda = np.trapezoid(tau_b_full, L_full)

    n_z = np.zeros_like(z_array)
    for i, z in enumerate(z_array):
        B_local = B_z(z, R_M, k, w)
        upper_limit = 1.0 / B_local
        
        if upper_limit <= 1.0 / R_M:
            n_z[i] = 0.0 
            continue
            
        u_max = np.sqrt(1.0 - B_local / R_M)
        
        def integrand_u(u):
            L = (1.0 - u**2) / B_local
            return 2.0 * I_1_interp(L)
            
        num, _ = quad(integrand_u, 0, u_max, limit=100) 
        
        # FIX 2: Divide by 2 to align the phase-space average with the physical spatial average
        n_z[i] = (avg_1_Lambda * num / denominator) / 2.0
        
    return n_z
# ==========================================
# 2. MASTER EXECUTION & PLOTTING
# ==========================================

R_M = 16.0
k = 1.3
w_values = [0.5, 0.3, 0.2, 0.1, 0.03]
colors = ['blue', 'red', 'limegreen', 'magenta', 'black']

z_array = np.linspace(0, 1, 200)
Lambda_arr = np.linspace(0, 0.999, 300)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
ax_B, ax_eta = axs[0, 0], axs[0, 1]
ax_I1, ax_n = axs[1, 0], axs[1, 1]

for w, color in zip(w_values, colors):
    print(f"Processing w = {w}...")
    
    # 1. Magnetic Field B(z)
    B = B_z(z_array, R_M, k, w)
    ax_B.plot(z_array, B, color=color, linewidth=2, label=f'w={w}')
    
    # 2. Pitch Angle Variable eta(Lambda)
    tau_b_arr = np.array([compute_tau_b(L, R_M, k, w) for L in Lambda_arr])
    integral_tau_b = cumulative_trapezoid(tau_b_arr, Lambda_arr, initial=0)
    eta_arr_full = 1.0 - (integral_tau_b / integral_tau_b[-1])
    ax_eta.plot(Lambda_arr, eta_arr_full, color=color, linewidth=2)
    
    # 3. First Eigenfunction I_1
    L_grid, I_1 = solve_first_eigenfunction(R_M, k, w)
    # Map Lambda grid to eta grid for plotting I_1
    tau_b_grid = np.array([compute_tau_b(L, R_M, k, w) for L in L_grid])
    integral_tau_b_grid = cumulative_trapezoid(tau_b_grid, L_grid, initial=0)
    eta_grid = 1.0 - (integral_tau_b_grid / integral_tau_b_grid[-1])
    ax_I1.plot(eta_grid, I_1, color=color, linewidth=2)
    
    # 4. Density Profile n(z)/<n>
    n_z = compute_density_profile(z_array, R_M, k, w, L_grid, I_1)
    ax_n.plot(z_array, n_z, color=color, linewidth=2)

# --- Formatting ---

# Panel a: B(z)
ax_B.set(xlim=(0, 1), ylim=(0, 16), xticks=[0, 0.5, 1], yticks=[0, 5, 10, 15], 
         xlabel='$z/L$', ylabel='$B/B_0$')
ax_B.text(0.05, 0.95, f'a)  $B(z)$\n    $R_M={int(R_M)}$', transform=ax_B.transAxes, va='top')
ax_B.legend(loc='upper right', frameon=False, fontsize=10)

# Panel b: eta(Lambda)
ax_eta.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], 
           xlabel='$\\Lambda$', ylabel='$\\eta$')
ax_eta.axvline(x=1.0/R_M, color='black', linestyle='--', linewidth=1)
ax_eta.text(1.0/R_M + 0.02, 0.05, '$1/R_M$', fontsize=11, va='bottom')
ax_eta.text(0.05, 0.95, 'b)  $\\eta(\\Lambda)$', transform=ax_eta.transAxes, va='top')

# Panel c: I_1(eta)
ax_I1.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], 
          xlabel='$\\eta$', ylabel='First Eigenfunction')
ax_I1.text(0.05, 0.95, 'c)  $I_1(\\eta)$', transform=ax_I1.transAxes, va='top')

# Panel e: Density
ax_n.set(xlim=(0, 1), ylim=(0, 2), xticks=[0, 0.5, 1], yticks=[0, 1, 2], 
         xlabel='$z/L$', ylabel='$n / \\langle n \\rangle$')
ax_n.text(0.05, 0.95, 'e)  Density profiles', transform=ax_n.transAxes, va='top')

plt.tight_layout()
plt.savefig('core_gemini.png', dpi=300, bbox_inches='tight')