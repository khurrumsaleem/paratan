"""
plot_fig3.py
============
Reproduces Figure 3 from Egedal et al. 2022.
Beam ion distribution function f(v, xi) from equation (14).

Run from inside the fbis/ directory.
Requires: mirror_geometry.py, eigenfunctions.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from eigenfunctions import find_eigenvalues, eval_M
from mirror_geometry import xi_TP

# ── Parameters matching Figure 3 in the paper ─────────────────────────────────
E_beam_keV = 60.0     # beam energy [keV]
Te_keV     = 1.0      # electron temperature [keV]
RM         = 13.3     # mirror ratio
theta_inj  = 45.0     # injection angle [degrees]
beta_m     = 1.0      # z_eff * mi / (2*mf) ~ 1 for D on D

# Critical energy Ec ~ 14.8 * Te for D beam into D plasma (eq. 3 + 16)
Ec_keV     = 14.8 * Te_keV
vc_over_v0 = np.sqrt(Ec_keV / E_beam_keV)   # vc/v0

xi_tp = xi_TP(RM)
xi0   = np.cos(np.radians(theta_inj))        # injection pitch angle = 1/sqrt(2)

print(f"vc/v0  = {vc_over_v0:.4f}")
print(f"xi_TP  = {xi_tp:.4f}")
print(f"xi0    = {xi0:.4f}")

# ── Compute eigenfunctions ─────────────────────────────────────────────────────
print("Computing eigenfunctions...")
eigens = find_eigenvalues(RM, n_modes=6)

# ── Source coefficients S_j  (eq. 11, delta source at xi=xi0) ─────────────────
# S_j = M_j(xi0) / (4*pi*alpha_j)
# We drop the 4*pi factor since it cancels in the normalised plot
Sj = np.array([
    eval_M(e, xi0) / e["alpha"]
    for e in eigens
])
print(f"S_j (relative): {np.round(Sj, 4)}")

# ── Build f(v, xi) on a grid  (eq. 14) ────────────────────────────────────────
# f(v,xi) = tau_s/(v^3 + vc^3) * sum_j S_j * M_j(xi) * u^lambda_j
# u = [(1 + vc^3)/(v^3 + vc^3) * v^3]^(beta_m/3)   [in units where v0=1]
# We plot log10(f * v0^3 / tau_s) so tau_s and v0 drop out.

Nv   = 400
Nxi  = 400
v_arr   = np.linspace(0.005, 1.0, Nv)    # v/v0,  y-axis
xi_arr  = np.linspace(0.0,   1.0, Nxi)   # xi_m,  x-axis

vc3 = vc_over_v0**3

# Precompute M_j on xi grid — zero outside loss cone
Mj_xi = np.zeros((len(eigens), Nxi))
for j, e in enumerate(eigens):
    Mj_xi[j, :] = eval_M(e, xi_arr)   # eval_M already zeros outside xi_TP

# Build distribution on (Nv, Nxi) grid
F = np.zeros((Nv, Nxi))
for iv, v in enumerate(v_arr):
    v3 = v**3
    u  = ((1.0 + vc3) / (v3 + vc3) * v3) ** (beta_m / 3.0)
    for j, e in enumerate(eigens):
        F[iv, :] += Sj[j] * Mj_xi[j, :] * u ** e["lam"]
    F[iv, :] /= (v3 + vc3)

# Zero out loss cone and v > v0 (solution only valid for v < v0)
F[:, xi_arr > xi_tp] = 0.0
F[v_arr >= 1.0, :]   = 0.0   # f=0 for v >= v0 (eq 14 only valid below injection speed)

# Normalise so peak ~ 15 to match paper colorbar
F_norm = F / np.nanmax(F) * 15.0

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))

im = ax.pcolormesh(xi_arr, v_arr, F_norm,
                   cmap='inferno', vmin=0, vmax=15,
                   shading='auto')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$f \cdot v_0^3 / \tau_s$ (normalised)', fontsize=11)
cbar.set_ticks([0, 5, 10, 15])

# Loss cone boundary
ax.axvline(xi_tp, color='white', lw=1.2, ls='--', alpha=0.8,
           label=fr'loss cone  $\xi_{{TP}}={xi_tp:.3f}$')

# Injection point
ax.plot(xi0, 1.0, 'o', color='cyan', ms=6, zorder=5,
        label=fr'injection  $\xi_0={xi0:.3f}$')

ax.set_xlabel(r'$\xi_m = v_\parallel/v$', fontsize=13)
ax.set_ylabel(r'$v/v_0$', fontsize=13)
ax.set_title(
    fr'Figure 3 — $f(v,\xi)$,  {E_beam_keV:.0f} keV beam,  '
    fr'{theta_inj:.0f}$^\circ$ injection,  '
    fr'$T_e={Te_keV:.0f}$ keV,  $R_M={RM}$',
    fontsize=10
)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.02)
ax.legend(fontsize=9, loc='lower left', framealpha=0.75)

fig.tight_layout()
fig.savefig('fig3_distribution.png', dpi=150, bbox_inches='tight')
print("Saved fig3_distribution.png")
plt.show()