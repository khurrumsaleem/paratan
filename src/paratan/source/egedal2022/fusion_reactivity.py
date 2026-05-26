"""
fusion_reactivity.py
====================
DT fusion reactivity and normalised fusion reactivity for Figure 4.
Egedal et al. 2022, eq. (17).

Bosch & Hale 1992, Nucl. Fusion 32, 611 — Table VII, eqs 12-14.

Public API
----------
sigma_v_DT(Ti_keV)              -> <v*sigma>_DT  [m^3/s]
sigma_ii(Ti_eV)                 -> Coulomb cross-section [m^2]
normalized_reactivity(Ti_keV)   -> log10 of RHS of eq.(17), Figure 4 y-axis
normalized_reactivity_linear(Ti_keV) -> linear RHS of eq.(17)
"""

import numpy as np

# Physical constants
_e    = 1.60217663e-19    # C
_eps0 = 8.85418781e-12    # F/m
_mp   = 1.67262192e-27    # kg
_mi   = 2.5 * _mp         # kg, average DT ion mass

DELTA_E_FUSION_eV  = 22.4e6    # eV  (17.6 MeV + 4.8 MeV T-breeding)
DELTA_E_FUSION_keV = 22400.0   # keV


# ── Bosch-Hale 1992, Table VII, T(d,n)^4He reactivity ────────────────────────
# eqs 12-14, valid 0.2-100 keV

_C1, _C2, _C3 = 1.17302e-9, 1.51361e-2, 7.51886e-2
_C4, _C5, _C6, _C7 = 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5
_BG   = 34.3827
_mrc2 = 1124656.0   # keV


def sigma_v_DT(Ti_keV):
    """
    DT fusion reactivity <v*sigma>_DT [m^3/s].
    Bosch-Hale 1992 Table VII.  Valid 0.2-100 keV.
    """
    T     = np.asarray(Ti_keV, dtype=float)
    theta = T / (1.0 - (T * (_C2 + T * (_C4 + T * _C6)))
                      / (1.0 + T * (_C3 + T * (_C5 + T * _C7))))
    xi    = (_BG**2 / (4.0 * theta)) ** (1.0 / 3.0)
    sv    = _C1 * theta * np.sqrt(xi / (_mrc2 * T**3)) * np.exp(-3.0 * xi)
    return sv * 1e-6   # cm^3/s -> m^3/s


# ── Coulomb pitch-angle scattering cross-section ──────────────────────────────
# sigma_ii = pi * e^4 * lnL / ((4*pi*eps0)^2 * T^2)
# where T is in Joules.

def sigma_ii_cross_section(Ti_eV, ln_lambda=17.0):
    """
    90-degree Coulomb scattering cross-section sigma_ii [m^2].
    sigma_ii = pi * e^4 * lnL / ((4*pi*eps0)^2 * Ti^2)
    Ti in eV.
    """
    Ti_J = np.asarray(Ti_eV, dtype=float) * _e
    num  = np.pi * _e**4 * ln_lambda
    den  = (4 * np.pi * _eps0)**2 * Ti_J**2
    return num / den


# ── Normalised reactivity — eq. (17) RHS — Figure 4 ─────────────────────────
# R = (<ov>_DT * E_fusion) / (4 * v_ti * sigma_ii * Ti)
# E_fusion and Ti both in eV -> units cancel.
# v_ti = sqrt(2*Ti/mi)  [m/s]
# Figure 4 plots log10(R) on y-axis.

def normalized_reactivity_linear(Ti_keV, ln_lambda=17.0):
    """
    Linear normalised fusion reactivity R(Ti) — RHS of eq. (17).
    Peak ~ 3.6 at Ti ~ 140 keV in linear scale.
    """
    Ti_keV = np.asarray(Ti_keV, dtype=float)
    Ti_eV  = Ti_keV * 1e3

    sv_dt  = sigma_v_DT(Ti_keV)                        # m^3/s
    sig_ii = sigma_ii_cross_section(Ti_eV, ln_lambda)   # m^2
    Ti_J   = Ti_eV * _e
    v_ti   = np.sqrt(2.0 * Ti_J / _mi)                 # m/s

    return (sv_dt * DELTA_E_FUSION_eV) / (4.0 * v_ti * sig_ii * Ti_eV)


def normalized_reactivity(Ti_keV, ln_lambda=17.0):
    """
    log10 of normalised fusion reactivity — y-axis of Figure 4.
    """
    return np.log10(normalized_reactivity_linear(Ti_keV, ln_lambda))