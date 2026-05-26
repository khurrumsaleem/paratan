"""
plot_figs1_2.py
===============
Reproduces Figures 1 and 2 from Egedal et al. 2022.

Figure 1 : Eigenfunctions M_lambda_j(xi) for j=1..5, RM=13.3
Figure 2a: Lowest eigenvalue lambda_1 vs 1/ln(RM) with analytic fit
Figure 2b: First eigenfunction M_1(xi) for various RM values
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mirror_geometry import xi_TP
from eigenfunctions import find_eigenvalues, lambda1_approx, compute_M

plt.rcParams.update({'font.size': 11, 'axes.linewidth': 0.8})
COLORS = ['#1f4e8c', '#d65108', '#2a7a2a', '#9b1d6b', '#6b4c11', '#1a8a8a']


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Eigenfunctions M_j(xi), RM = 13.3
# ══════════════════════════════════════════════════════════════════════════════

RM_fig1 = 13.3
print(f"Computing eigenfunctions for RM = {RM_fig1} ...")
eigens_13 = find_eigenvalues(RM_fig1, n_modes=5)

print("\nTable 1 comparison (RM = 13.3):")
print(f"{'j':>3}  {'l_j':>6}  {'lambda_j':>10}  {'alpha_j':>8}")
paper_vals = [
    (0.56, 0.88,  0.66),
    (3.07, 12.5,  0.64),
    (5.52, 36.0,  0.64),
    (7.96, 71.0,  0.64),
    (10.40, 118.0, 0.64),
]
for i, e in enumerate(eigens_13):
    pv = paper_vals[i] if i < len(paper_vals) else ("—","—","—")
    print(f"  {e['j']:>1}  computed: lj={e['lj']:6.3f}  lam={e['lam']:8.2f}  alpha={e['alpha']:.3f}"
          f"   paper: lj={pv[0]}  lam={pv[1]}  alpha={pv[2]}")

fig1, ax1 = plt.subplots(figsize=(5.5, 4))
xi_tp_13 = xi_TP(RM_fig1)

for i, e in enumerate(eigens_13):
    ax1.plot(e["xi"], e["M"], color=COLORS[i], lw=1.6, label=f'j={e["j"]}')

ax1.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
ax1.axvline(xi_tp_13, color='gray', lw=0.8, ls=':', alpha=0.6)
ax1.text(xi_tp_13 + 0.005, 1.35, r'$\xi_{TP}$', color='gray', fontsize=10)

ax1.set_xlabel(r'$\xi$', fontsize=13)
ax1.set_ylabel(r'$M_{\lambda_j}(\xi)$', fontsize=13)
ax1.set_xlim(0, 1.0)
ax1.set_ylim(-1.6, 1.55)
ax1.legend(fontsize=10, loc='upper right', framealpha=0.8)
ax1.set_title(fr'Fig 1 — Eigenfunctions, $R_M = {RM_fig1}$', fontsize=11)
fig1.tight_layout()
fig1.savefig('fig1_eigenfunctions.png', dpi=150)
print("\nSaved: fig1_eigenfunctions.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2a — lambda_1 vs 1/ln(RM)
# ══════════════════════════════════════════════════════════════════════════════

# RM values to sample (paper marks RM = 2, 4, 8, 32, 128, 1024)
RM_marked = [2, 4, 8, 32, 128, 1024]
RM_scan   = np.logspace(np.log10(1.5), np.log10(2000), 60)

print("\nComputing lambda_1 scan over RM ...")
lam1_numerical = []
for RM_ in RM_scan:
    try:
        e = find_eigenvalues(RM_, n_modes=1)
        lam1_numerical.append(e[0]["lam"])
    except Exception as ex:
        print(f"  RM={RM_:.1f} failed: {ex}")
        lam1_numerical.append(np.nan)
lam1_numerical = np.array(lam1_numerical)

inv_lnRM_scan = 1.0 / np.log(RM_scan)
lam1_analytic = lambda1_approx(RM_scan)

fig2, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax2a = axes[0]
ax2a.plot(inv_lnRM_scan, lam1_numerical, 'k-',  lw=1.8, label='numerical')
ax2a.plot(inv_lnRM_scan, lam1_analytic,  'g--', lw=1.4,
          label=r'$2/\ln(R_M)+0.37(1/\ln R_M)^{1.3}$')

# Mark specific RM values
for RM_ in RM_marked:
    x_ = 1.0 / np.log(RM_)
    y_ = lambda1_approx(RM_)
    ax2a.plot(x_, y_, 'r*', ms=9, zorder=5)
    ax2a.annotate(str(RM_), (x_, y_), textcoords='offset points',
                  xytext=(4, 4), fontsize=8, color='red')

ax2a.set_xlabel(r'$1/\ln(R_M)$', fontsize=13)
ax2a.set_ylabel(r'$\lambda_1$', fontsize=13)
ax2a.set_xlim(0, 1.5)
ax2a.set_ylim(0, 4.0)
ax2a.legend(fontsize=9)
ax2a.set_title('Fig 2a — Lowest eigenvalue', fontsize=11)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2b — First eigenfunction M_1(xi) for various RM
# ══════════════════════════════════════════════════════════════════════════════

RM_list_2b = [2, 4, 8, 32, 128, 1024]
ax2b = axes[1]

xi_common = np.linspace(0, 1.0, 600)
for i, RM_ in enumerate(RM_list_2b):
    xi_tp_ = xi_TP(RM_)
    xi_grid_ = np.linspace(0.0, xi_tp_ * (1 - 1e-7), 600)
    lam1_ = find_eigenvalues(RM_, n_modes=1)[0]["lam"]
    M1_   = compute_M(lam1_, xi_grid_)
    # extend with zeros beyond xi_TP for plotting to xi=1
    xi_full = np.concatenate([xi_grid_, [xi_tp_, 1.0]])
    M_full  = np.concatenate([M1_,      [0.0,    0.0]])
    ax2b.plot(xi_full, M_full, color=COLORS[i % len(COLORS)],
              lw=1.5, label=fr'$R_M={RM_}$')

ax2b.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
ax2b.set_xlabel(r'$\xi$', fontsize=13)
ax2b.set_ylabel(r'$M_{\lambda_1}(\xi)$', fontsize=13)
ax2b.set_xlim(0, 1.0)
ax2b.set_ylim(-0.05, 1.05)
ax2b.legend(fontsize=9, loc='lower left')
ax2b.set_title(r'Fig 2b — First eigenfunction vs $R_M$', fontsize=11)

fig2.tight_layout()
fig2.savefig('fig2_eigenvalues.png', dpi=150)
print("Saved: fig2_eigenvalues.png")

plt.savefig('fig2_eigenvalues.png', dpi=150)