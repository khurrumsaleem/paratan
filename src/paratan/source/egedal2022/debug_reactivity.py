"""
debug_reactivity.py
Print intermediate values at Ti=10, 64, 140 keV to find where it breaks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# copy the params inline so we don't depend on the module
p = dict(
    BG   = 34.3827,
    mrc2 = 1124656.0,
    C1   =  1.17302e-9,
    C2   =  1.51361e-2,
    C3   =  7.51886e-2,
    C4   =  4.60643e-3,
    C5   =  1.35000e-2,
    C6   = -1.06750e-4,
    C7   =  1.36600e-5,
)

for T in [1.0, 10.0, 64.0, 140.0]:
    num   = T * (p["C2"] + T * (p["C4"] + T * p["C6"]))
    denom = 1.0 + T * (p["C3"] + T * (p["C5"] + T * p["C7"]))
    frac  = num / denom
    theta = T / (1.0 - frac)
    print(f"T={T:6.1f} keV | num={num:.6f} | denom={denom:.6f} | "
          f"frac={frac:.6f} | theta={theta:.4f}")
    if theta > 0:
        xi  = (p["BG"]**2 / (4.0 * theta)) ** (1.0/3.0)
        sv  = p["C1"] * theta * np.sqrt(xi / (p["mrc2"] * T**3)) * np.exp(-3.0 * xi)
        print(f"           | xi={xi:.4f} | sv={sv:.4e} cm3/s = {sv*1e-6:.4e} m3/s")
    else:
        print(f"           | theta <= 0, skipped")

# also check sigma_v_ii at same temps
print()
for T in [1.0, 10.0, 64.0, 140.0]:
    sv_ii = 6.27e-15 * 15.0 / T**1.5
    print(f"T={T:6.1f} keV | sigma_v_ii = {sv_ii:.4e} m3/s")

# sanity check sigma_v_ii from SI directly
print("\n--- sigma_v_ii from SI ---")
e    = 1.6e-19
eps0 = 8.854e-12
mi   = 2.5 * 1.673e-27
import numpy as np
for T_keV in [1.0, 10.0, 64.0, 140.0]:
    Ti_J  = T_keV * 1.6e-16
    v_ti  = np.sqrt(2.0 * Ti_J / mi)
    kC2   = (e**2 / (4*np.pi*eps0))**2
    coeff = (4/3)*np.sqrt(2/np.pi)
    sv_ii = coeff * kC2 * 15.0 / (mi**2 * v_ti**3)
    print(f"T={T_keV:6.1f} keV | v_ti={v_ti:.3e} m/s | sv_ii={sv_ii:.4e} m3/s")
from fusion_reactivity import sigma_v_DT, sigma_v_ii, normalized_reactivity, DELTA_E_FUSION_keV
for T in [10.0, 64.0, 140.0]:
    sv_dt = sigma_v_DT(T)
    sv_ii = sigma_v_ii(T)
    R     = normalized_reactivity(T)
    print(f"T={T:6.1f} keV | sv_DT={sv_dt:.4e} | sv_ii={sv_ii:.4e} | R={R:.4f}")