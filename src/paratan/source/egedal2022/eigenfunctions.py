"""
eigenfunctions.py
=================
Eigenfunctions M_lambda(xi) of the Lorentz pitch-angle scattering operator
for a square magnetic mirror.  Egedal et al. 2022, section 2.2.

The operator is:
    L[M] = d/dxi [(1-xi^2) dM/dxi] = -lambda * M

with boundary conditions:
    M(0)    = 1      (normalised, even solution)
    M'(0)   = 0      (even => zero derivative at xi=0)
    M(xi_TP) = 0     (loss-cone BC selects discrete eigenvalues lambda_j)

Public API
----------
compute_M(lam, xi_grid)          -> array   integrate M_lam on xi_grid
find_eigenvalue(j, RM, lam_guess) -> float   root-find lambda_j
find_eigenvalues(RM, n_modes)    -> list[dict]  full table {lj, lam, alpha}
norm_alpha(M_vals, xi_grid)      -> float   orthonormality factor alpha_j (eq.8)

The returned eigenvalue dicts match Table 1 of the paper:
    {"lj": l_j, "lam": lambda_j, "alpha": alpha_j}
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from mirror_geometry import xi_TP as _xi_TP


# ── Low-level ODE integrator ───────────────────────────────────────────────────

def compute_M(lam, xi_grid):
    """
    Numerically integrate M_lambda(xi) on xi_grid.

    Solves:  (1-xi^2) M'' - 2*xi*M' + lambda*M = 0
    ICs:     M(0) = 1,  M'(0) = 0   (even, normalised)

    Parameters
    ----------
    lam      : float  eigenvalue lambda
    xi_grid  : 1-D array, must start at (or very near) 0

    Returns
    -------
    M : 1-D array, same length as xi_grid
    """
    def ode(xi, y):
        M, dM = y
        denom = 1.0 - xi * xi
        d2M = (2.0 * xi * dM - lam * M) / denom if abs(denom) > 1e-14 else 0.0
        return [dM, d2M]

    sol = solve_ivp(
        ode,
        t_span=(xi_grid[0], xi_grid[-1]),
        y0=[1.0, 0.0],
        t_eval=xi_grid,
        method='RK45',
        rtol=1e-10,
        atol=1e-13,
        dense_output=False,
    )
    return sol.y[0]


# ── Eigenvalue root-finder ─────────────────────────────────────────────────────

def _M_at_xi_TP(lam, xi_TP_val, n_pts=800):
    """Evaluate M_lambda at the loss-cone boundary xi_TP."""
    xi_grid = np.linspace(0.0, xi_TP_val * (1.0 - 1e-7), n_pts)
    M = compute_M(lam, xi_grid)
    return M[-1]


def find_eigenvalue(j_index, RM, lam_bracket=None, tol=1e-10):
    """
    Find the j-th eigenvalue lambda_j for mirror ratio RM.

    The j=1 eigenvalue is near   2/ln(RM) + 0.37*(1/ln(RM))^1.3  (eq. after fig 2a).
    Higher eigenvalues are spaced roughly as (2j-1)^2 * lambda_1.

    Parameters
    ----------
    j_index     : int  1-based mode index
    RM          : float mirror ratio
    lam_bracket : (lam_lo, lam_hi) search bracket; auto-estimated if None

    Returns
    -------
    lam_j : float
    """
    xi_tp = _xi_TP(RM)
    lnRM  = np.log(RM)

    if lam_bracket is None:
        # Rough estimate for the j-th zero
        lam1_approx = 2.0 / lnRM + 0.37 * (1.0 / lnRM) ** 1.3
        if j_index == 1:
            lo, hi = lam1_approx * 0.5, lam1_approx * 2.0
        else:
            # Higher modes scale roughly as l_j*(l_j+1) where l_j grows ~linearly
            lo = lam1_approx * (2 * j_index - 2) ** 2 + 0.1
            hi = lam1_approx * (2 * j_index + 1) ** 2 + 1.0
    else:
        lo, hi = lam_bracket

    # Make sure lo > 0
    lo = max(lo, 1e-4)

    f = lambda lam: _M_at_xi_TP(lam, xi_tp)

    # Scan for sign change if initial bracket doesn't work
    if np.sign(f(lo)) == np.sign(f(hi)):
        lam_scan = np.linspace(lo, hi, 2000)
        vals = [f(l) for l in lam_scan]
        bracket_found = False
        for k in range(len(vals) - 1):
            if vals[k] * vals[k + 1] < 0:
                lo, hi = lam_scan[k], lam_scan[k + 1]
                bracket_found = True
                break
        if not bracket_found:
            raise RuntimeError(
                f"Could not bracket eigenvalue j={j_index} for RM={RM}. "
                f"Try supplying lam_bracket manually."
            )

    lam_j = brentq(f, lo, hi, xtol=tol, rtol=tol)
    return lam_j


def norm_alpha(M_vals, xi_grid):
    """
    Compute normalisation factor alpha_j = integral_0^xi_TP M_j^2 dxi  (eq. 8).
    Uses numpy trapezoid integration.
    """
    return np.trapezoid(M_vals ** 2, xi_grid)


# ── Full eigenvalue table ──────────────────────────────────────────────────────

def find_eigenvalues(RM, n_modes=6, n_pts=800):
    """
    Compute the first n_modes eigenvalues and eigenfunctions for mirror ratio RM.

    Returns
    -------
    eigens : list of dicts, each containing:
        "lj"    : float  order l_j  (lambda_j = l_j*(l_j+1))
        "lam"   : float  eigenvalue lambda_j
        "alpha" : float  normalisation factor alpha_j
        "M"     : 1-D array, M_j(xi) evaluated on xi_grid
        "xi"    : 1-D array, the xi grid used
    """
    xi_tp   = _xi_TP(RM)
    xi_grid = np.linspace(0.0, xi_tp * (1.0 - 1e-7), n_pts)
    lnRM    = np.log(RM)
    lam1_approx = 2.0 / lnRM + 0.37 * (1.0 / lnRM) ** 1.3

    eigens = []
    for j in range(1, n_modes + 1):
        if j == 1:
            bracket = (lam1_approx * 0.3, lam1_approx * 3.0)
        else:
            # Modes are spaced roughly evenly in l-space with dl ~ 2.45.
            # For j=2 we don't have two points yet so use the known spacing.
            # For j>=3 we extrapolate from the previous two computed modes.
            prev_lj = eigens[-1]["lj"]
            if j >= 3:
                dl = eigens[-1]["lj"] - eigens[-2]["lj"]
            else:
                dl = 2.45   # universal approximate l-spacing between mirror modes
            next_lj_lo = prev_lj + dl * 0.6
            next_lj_hi = prev_lj + dl * 1.5
            lo = max(next_lj_lo * (next_lj_lo + 1), eigens[-1]["lam"] + 1.0)
            hi = next_lj_hi * (next_lj_hi + 1)
            bracket = (lo, hi)

        lam_j = find_eigenvalue(j, RM, lam_bracket=bracket)
        lj    = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * lam_j))  # l(l+1)=lam => l
        M     = compute_M(lam_j, xi_grid)
        alpha = norm_alpha(M, xi_grid)

        eigens.append({
            "j":     j,
            "lj":    lj,
            "lam":   lam_j,
            "alpha": alpha,
            "M":     M,
            "xi":    xi_grid,
        })

    return eigens


# ── Convenience: interpolate M_j at arbitrary xi ──────────────────────────────

def eval_M(eigen, xi):
    """
    Interpolate eigenfunction M_j at xi (scalar or array).
    Returns 0 outside [0, xi_TP].
    """
    xi = np.asarray(xi)
    xi_tp = eigen["xi"][-1]
    result = np.where(
        (xi >= 0) & (xi <= xi_tp),
        np.interp(xi, eigen["xi"], eigen["M"], left=1.0, right=0.0),
        0.0
    )
    return result


# ── Analytic approximation for lambda_1 ───────────────────────────────────────

def lambda1_approx(RM):
    """
    Analytic approximation for lowest eigenvalue (eq. after fig 2a):
        lambda_1 ≈ 2/ln(RM) + 0.37*(1/ln(RM))^1.3
    """
    x = 1.0 / np.log(RM)
    return 2.0 * x + 0.37 * x ** 1.3
