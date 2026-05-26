"""
mirror_geometry.py
==================
Mirror geometry primitives for the FBIS solver.
Based on Egedal et al. 2022 (Nucl. Fusion 62, 126053).

Public API
----------
xi_TP(RM)               — trapped/passing boundary (eq. after eq. 5)
B_square(z, RM, L)      — ideal square-well field profile (eq. 5)
B_smooth(z, RM, w, k)   — smooth realistic profile (eq. 56)
"""

import numpy as np


def xi_TP(RM):
    """
    Trapped/passing pitch-angle boundary for a square mirror.
    xi_TP = sqrt(1 - 1/RM)
    Particles with |xi| < xi_TP are magnetically trapped.
    """
    return np.sqrt(1.0 - 1.0 / RM)


def B_square(z, RM, L=1.0):
    """
    Ideal square-well magnetic field (eq. 5).
    B = B0        for |z| <= L
    B = RM * B0   for |z| >  L
    Returns B/B0 (normalised).
    """
    z = np.asarray(z)
    return np.where(np.abs(z) <= L, 1.0, float(RM))


def B_smooth(z, RM, w=0.1, k=1.3, L=1.0):
    """
    Smooth realistic mirror field profile (eq. 56).
    B(z)/B0 = 1 + (RM-1) * [exp(-||z/L|-1|^k / w^k) - exp(-1/w^k)]
                          / [1 - exp(-1/w^k)]
    Parameters
    ----------
    z   : array-like, axial coordinate (normalised so mirror throat is at |z|=L)
    RM  : mirror ratio
    w   : width parameter (controls how square the profile is; w->0 => square well)
    k   : shape exponent (paper uses k=1.3)
    L   : half-length of central cell (default 1, so z is in units of L)
    Returns B/B0.
    """
    z = np.asarray(z, dtype=float)
    zn = np.abs(z) / L          # normalised |z|
    exp_denom = np.exp(-1.0 / w**k)
    num = np.exp(-np.abs(zn - 1.0)**k / w**k) - exp_denom
    denom = 1.0 - exp_denom
    return 1.0 + (RM - 1.0) * num / denom