"""
reduced_model.py
================
Reduced analytical model for a pure beam plasma in a square mirror.
Egedal et al. 2022, Section 3.

Implements equations (20)-(31) — the semi-analytic expressions for:
  - f1(v):  approximate beam ion speed distribution  (eq. 20)
  - alpha:  dimensionless drag parameter             (eq. 21)
  - Ti_tilde(alpha):  normalised ion temperature     (eq. 22)
  - H(alpha):  density integral function             (eq. 23)
  - Te_tilde from power balance                      (eq. 24)
  - tau_P_tilde:  normalised confinement time        (eq. 29)

Public API
----------
alpha_param(Te_tilde, Ti_tilde, RM)   -> alpha  (eq. 21)
Ti_tilde_from_alpha(alpha)            -> Ti/Ebeam  (eq. 22)
H_func(alpha)                         -> H(alpha)  (eq. 23)
Te_tilde_from_alpha(alpha, RM)        -> Te/Ebeam  (inverted from eq. 21)
solve_temperatures(RM, p_aux)         -> (Te_tilde, Ti_tilde) self-consistent pair
tau_P_tilde(RM, Ti_tilde, eigens)     -> normalised confinement time  (eq. 29)
f1_normalised(v_hat, alpha)           -> normalised f1(v/v0)  (eq. 20)
"""

import numpy as np
from scipy.special import gammaincc, gamma
from scipy.optimize import brentq


# ── alpha parameter  (eq. 21) ─────────────────────────────────────────────────
# alpha = v_c^3 * beta_m * lambda_1 / (4 * v0^2 * v_ti)
#       ~ 22.4 * Te_tilde^(3/2) / (Ti_tilde^(1/2) * ln(RM))
# where Te_tilde = Te/Ebeam,  Ti_tilde = Ti/Ebeam,  lambda_1 ~ 2/ln(RM)

def alpha_param(Te_tilde, Ti_tilde, RM):
    """
    Dimensionless drag parameter alpha (eq. 21).
    alpha ~ 22.4 * Te_tilde^1.5 / (Ti_tilde^0.5 * ln(RM))
    """
    return 22.4 * Te_tilde**1.5 / (np.sqrt(Ti_tilde) * np.log(RM))


# ── Ti_tilde from alpha  (eq. 22) ─────────────────────────────────────────────
# (3/2) * Ti_tilde = [exp(-alpha) - Gamma(0,alpha)] / Gamma(0,alpha)
# where Gamma(0,alpha) is the upper incomplete gamma function

def _Gamma0(alpha):
    """Upper incomplete gamma function Gamma(0, alpha) = integral_alpha^inf t^-1 e^-t dt."""
    # scipy: gammaincc(a,x) = Gamma(a,x)/Gamma(a), so Gamma(0,x) needs care.
    # Use: Gamma(0,x) = -Ei(-x) = expn(1,x)  or directly via scipy.special.exp1
    from scipy.special import exp1
    return exp1(alpha)


def Ti_tilde_from_alpha(alpha):
    """
    Normalised ion temperature Ti/Ebeam as a function of alpha (eq. 22).
    (3/2) * Ti_tilde = [exp(-alpha) - Gamma(0,alpha)] / Gamma(0,alpha)
    """
    alpha = np.asarray(alpha, dtype=float)
    G0    = _Gamma0(alpha)
    return (2.0 / 3.0) * (np.exp(-alpha) - G0) / G0


# ── H function  (eq. 23) ──────────────────────────────────────────────────────
# H(alpha) = alpha * exp(-alpha) / Gamma(0, alpha)
# -> 1 as alpha -> 0  (weak drag limit)
# -> alpha as alpha -> inf  (strong drag, ions slow to stop)

def H_func(alpha):
    """
    H(alpha) = alpha * exp(-alpha) / Gamma(0, alpha)   (eq. 23)
    Used in computing beam density and confinement time.
    """
    alpha = np.asarray(alpha, dtype=float)
    G0    = _Gamma0(alpha)
    return alpha * np.exp(-alpha) / G0


# ── Te_tilde from alpha  (eq. 21 inverted) ────────────────────────────────────
# alpha = 22.4 * Te_tilde^1.5 / (Ti_tilde^0.5 * ln(RM))
# Given alpha and Ti_tilde (which itself depends on alpha), we need to
# solve self-consistently. The paper presents Te_tilde/ln(RM)^(2/3) as a
# function of alpha — we invert that relationship.

def Te_tilde_from_alpha(alpha, RM):
    """
    Te/Ebeam as a function of alpha and RM, from eq. (21) inverted:
        Te_tilde = (alpha * Ti_tilde^0.5 * ln(RM) / 22.4)^(2/3)
    where Ti_tilde is itself computed from alpha via eq. (22).
    """
    Ti_t = Ti_tilde_from_alpha(alpha)
    return (alpha * np.sqrt(Ti_t) * np.log(RM) / 22.4) ** (2.0 / 3.0)


# ── Self-consistent (Te_tilde, Ti_tilde) from power balance  (eq. 24) ─────────
# Power balance:  1 + p_aux = Ti_tilde + 6*Te_tilde
# Combined with Ti_tilde(alpha) and Te_tilde(alpha, RM), solve for alpha.

def solve_temperatures(RM, p_aux=0.0):
    """
    Find self-consistent (Te_tilde, Ti_tilde) by solving eq. (24):
        1 + p_aux = Ti_tilde(alpha) + 6 * Te_tilde(alpha, RM)

    Parameters
    ----------
    RM    : float, mirror ratio
    p_aux : float, auxiliary heating fraction (Paux/Ebeam), default 0

    Returns
    -------
    Te_tilde, Ti_tilde : floats
    alpha              : float
    """
    target = 1.0 + p_aux

    def residual(alpha):
        Ti_t = Ti_tilde_from_alpha(alpha)
        Te_t = Te_tilde_from_alpha(alpha, RM)
        if Ti_t <= 0 or Te_t <= 0:
            return 1e10
        return Ti_t + 6.0 * Te_t - target

    # Alpha scan: at very small alpha (weak drag) Ti_tilde -> 2/3,
    # Te_tilde -> 0, sum -> 2/3 < 1. At large alpha sum grows. Root is between.
    # Bracket: alpha in (1e-3, 100)
    try:
        alpha_sol = brentq(residual, 1e-4, 200.0, xtol=1e-10)
    except ValueError:
        raise RuntimeError(
            f"Could not find self-consistent alpha for RM={RM}, p_aux={p_aux}. "
            f"residual(1e-4)={residual(1e-4):.3f}, residual(200)={residual(200):.3f}"
        )

    Ti_t = Ti_tilde_from_alpha(alpha_sol)
    Te_t = Te_tilde_from_alpha(alpha_sol, RM)
    return Te_t, Ti_t, alpha_sol


# ── Normalised confinement time  (eq. 29) ─────────────────────────────────────
# tau_P_tilde = (1 / alpha_1*lambda_1) * H(alpha)/Ti_tilde * integral_0^1 M1 dxi

def tau_P_tilde(alpha, Ti_tilde, eigens):
    """
    Normalised confinement time tau_P / tau_90_Ti  (eq. 29).

    tau_P_tilde = (1 / alpha_1 * lambda_1) * H(alpha)/Ti_tilde * int_0^1 M1 dxi

    Parameters
    ----------
    alpha     : float, drag parameter
    Ti_tilde  : float, Ti/Ebeam
    eigens    : list of eigen dicts from find_eigenvalues() — needs j=1 entry

    Returns
    -------
    tau_tilde : float
    """
    e1       = eigens[0]   # j=1 mode
    alpha_1  = e1["alpha"]
    lambda_1 = e1["lam"]
    # integral of M1 over [0, xi_TP] — already on the stored xi grid
    int_M1   = np.trapezoid(e1["M"], e1["xi"])
    H        = H_func(alpha)
    return (1.0 / (alpha_1 * lambda_1)) * (H / Ti_tilde) * int_M1


# ── Normalised f1(v)  (eq. 20) ────────────────────────────────────────────────
# f1(v) = (tau_s * S1 / exp(-alpha)) * exp(-alpha * v0^2/v^2) / v^3   for v < v0
# Normalised form: f1_norm = v0^3 * f1 / (3 * integral v^2*f1 dv)

def f1_normalised(v_hat, alpha):
    """
    Normalised beam ion speed distribution (eq. 20), Figure 6.
        f1_norm(v/v0) = v0^3 * f1(v) / (3 * integral_0^v0 v^2*f1 dv)

    Parameters
    ----------
    v_hat : array, v/v0 values in [0, 1]
    alpha : float, drag parameter

    Returns
    -------
    f1_norm : array, same shape as v_hat
    """
    v_hat = np.asarray(v_hat, dtype=float)
    # unnormalised: f1 * v^3 / (tau_s*S1) = exp(-alpha) * exp(-alpha/v_hat^2) / v_hat^3 * v_hat^3
    # = exp(-alpha) * exp(-alpha * (1/v_hat^2 - 0))  -- just the v-dependent part
    # f1(v) propto exp(-alpha * v0^2/v^2) / v^3  for v<v0
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_raw = np.where(v_hat > 0, np.exp(-alpha / v_hat**2) / v_hat**3, 0.0)
        f1_raw = np.where(v_hat <= 1.0, f1_raw, 0.0)

    # Normalise: integral_0^1 v_hat^2 * f1_raw d(v_hat) gives density up to const
    norm = 3.0 * np.trapezoid(v_hat**2 * f1_raw, v_hat)
    if norm <= 0:
        return np.zeros_like(v_hat)
    return v_hat**3 * f1_raw / norm   # = v0^3 * f1 / (3 * int v^2 f1 dv)