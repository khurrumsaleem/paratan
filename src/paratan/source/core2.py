import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def B_egedal(z, R_M, w, k=1.3):
    """
    Egedal (2022) eq. 56 magnetic field profile for a mirror machine.
    
    Parameters
    ----------
    z    : float or array, position along mirror, normalized so -1 <= z <= 1
           (z=0 is midplane, z=±1 is mirror throat)
    R_M  : float, mirror ratio B_max / B_min
    w    : float, characteristic length scale near mirror throat
               w -> 0  approaches square well
               w ~ 0.1 is realistic lab config (paper's default)
               w ~ 0.2 is practical upper limit before confinement degrades
    k    : float, shape exponent (default 1.3 as used in paper)
    
    Returns
    -------
    B/B0 : normalized magnetic field (= 1 at midplane, = R_M at throat)
    """
    numerator   = np.exp(-( np.abs(np.abs(z) - 1) )**k / w**k) \
                - np.exp(-1.0 / w**k)
    denominator = 1.0 - np.exp(-1.0 / w**k)
    
    return 1.0 + (R_M - 1.0) * numerator / denominator

# def plot_B_egedal(R_M, w, k=1.3):
#     """
#     Plot the magnetic field profile for a mirror machine.
#     """
#     z = np.linspace(-1, 1, 100)
#     B = B_egedal(z, R_M, w, k)
#     plt.plot(z, B)
#     plt.xlabel('z')
#     plt.ylabel('B/B0')
#     plt.title('Magnetic Field Profile for a Mirror Machine')
#     plt.savefig(f'B_egedal_R_M_{R_M}_w_{w}_k_{k}.png')
    
def tau_b_normalized(Lambda, z_grid, B_norm):
    """
    Egedal (2022) eq. 37 — normalized bounce time.
    
    Parameters
    ----------
    Lambda  : float, pitch angle variable = mu*B0/E, range [1/R_M, 1]
    z_grid  : 1D array, z positions (normalized, 0 to 1 = midplane to throat)
    B_norm  : 1D array, B(z)/B0 on z_grid (from B_egedal)
    
    Returns
    -------
    tau_b_tilde : float, normalized bounce time (dimensionless, in [0, 1])
                  = 0 means particle spends no time in cell (passing)
                  = 1 means particle is stuck at midplane (deeply trapped)
    L_b         : float, turning point in z (normalized coords)
                  = NaN if particle is in loss cone (Lambda < 1/R_M)
    """
    from scipy.interpolate import interp1d
    
    R_M = B_norm[-1]  # max B/B0 at throat (z=1)
    
    # Loss cone check: particle is lost if Lambda < 1/R_M
    if Lambda < 1.0 / R_M:
        return 0.0, np.nan
    
    # Find turning point L_b: where B(z)*Lambda/B0 = 1
    # i.e. B_norm(z) = 1/Lambda
    B_at_turning = 1.0 / Lambda
    
    # Interpolate to find L_b
    B_interp = interp1d(z_grid, B_norm, kind='linear')
    
    # Find z where B_norm = B_at_turning (only exists if B_at_turning <= R_M)
    # Search between midplane and throat
    from scipy.optimize import brentq
    try:
        L_b = brentq(lambda z: B_interp(z) - B_at_turning, z_grid[0], z_grid[-1])
    except ValueError:
        # B never reaches 1/Lambda — particle is trapped at midplane (Lambda ~ 1)
        L_b = z_grid[-1]
    
    # Integrate v_parallel/v from 0 to L_b
    # v_parallel/v = sqrt(1 - B(z)*Lambda)
    v_par_over_v = lambda z: np.sqrt(np.maximum(1.0 - B_interp(z) * Lambda, 0.0))
    
    integral, _ = quad(v_par_over_v, z_grid[0], L_b, 
                       limit=200, epsabs=1e-8, epsrel=1e-8)
    
    # Normalize by l (half-length = 1 in normalized coords)
    l = z_grid[-1]  # = 1.0
    tau_tilde = integral / l
    
    return tau_tilde, L_b

def n_profile_normalized(Lambda_0, z_grid, B_norm):
    """
    Normalized collisionless beam ion density profile n(z)/<n>
    for a mono-energetic beam injected at fixed pitch angle Lambda_0.
    
    Parameters
    ----------
    Lambda_0 : float, pitch angle variable = sin^2(theta_inj)
               e.g. 0.5 for 45 deg, 1.0 for 90 deg injection
    z_grid   : 1D array, z positions normalized 0 (midplane) to 1 (throat)
    B_norm   : 1D array, B(z)/B0 on z_grid (from B_egedal)
    
    Returns
    -------
    n_norm   : 1D array, n(z)/<n>, same shape as z_grid
               peaked at turning point, zero beyond it
    L_b      : float, turning point location in normalized z
    """
    R_M = B_norm[-1]
    
    # Loss cone check
    if Lambda_0 < 1.0 / R_M:
        raise ValueError(f"Lambda_0={Lambda_0} is in the loss cone (< 1/R_M = {1/R_M:.3f})")
    
    B_interp = interp1d(z_grid, B_norm, kind='linear')
    
    # --- Step 1: find turning point L_b where B(z)*Lambda_0 = 1 ---
    B_at_turning = 1.0 / Lambda_0
    
    if B_at_turning >= R_M:
        # Particle turns exactly at or beyond throat — pin to throat
        L_b = z_grid[-1]
    else:
        L_b = brentq(lambda z: B_interp(z) - B_at_turning, 
                     z_grid[0] + 1e-6, z_grid[-1])
    
    # --- Step 2: compute tau_b_tilde (normalization) ---
    # integral of v_par/v = sqrt(1 - B(z)*Lambda_0) from 0 to L_b
    integrand = lambda z: np.sqrt(np.maximum(1.0 - B_interp(z) * Lambda_0, 0.0))
    tau_tilde, _ = quad(integrand, z_grid[0], L_b,
                        limit=200, epsabs=1e-8, epsrel=1e-8,
                        points=[L_b * 0.999])  # hint near singularity
    tau_tilde /= z_grid[-1]  # normalize by l=1
    
    # --- Step 3: compute n(z)/<n> = 1 / (tau_tilde * sqrt(1 - B(z)*Lambda_0)) ---
    n_norm = np.zeros_like(z_grid, dtype=float)
    
    mask = z_grid <= L_b
    B_masked = B_norm[mask]
    
    v_par_sq = np.maximum(1.0 - B_masked * Lambda_0, 0.0)
    
    # Avoid actual singularity at turning point — clip to small floor
    v_par_sq = np.maximum(v_par_sq, 1e-6)
    
    n_norm[mask] = 1.0 / (tau_tilde * np.sqrt(v_par_sq))
    
    return n_norm, L_b

