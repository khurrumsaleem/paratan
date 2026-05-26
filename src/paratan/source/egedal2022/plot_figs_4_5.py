"""
plot_fig4.py
============
Reproduces Figure 4 from Egedal et al. 2022.
Run from inside fbis/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from fusion_reactivity import normalized_reactivity, normalized_reactivity_linear

log_T_eV = np.linspace(3.9, 6.0, 500)
T_eV     = 10**log_T_eV
T_keV    = T_eV / 1e3

log_R = normalized_reactivity(T_keV)
R     = normalized_reactivity_linear(T_keV)

i_peak = np.argmax(R)
print(f"Peak R      = {R[i_peak]:.3f}")
print(f"Peak log10R = {log_R[i_peak]:.3f}")
print(f"At Ti       = {T_keV[i_peak]:.1f} keV")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(log_T_eV, log_R, color='blue', lw=2.5)

ax.set_xlim(3.9, 6.0)
ax.set_ylim(-1.5, 0.8)
ax.set_xlabel(r'$\log_{10}(T_i/\mathrm{eV})$', fontsize=13)
ax.set_ylabel(r'$\log_{10}\!\left(\frac{\langle v\sigma\rangle_{DT}\,\Delta E_F}{4\,v_{ti}\sigma^{ii}_{ti}\,T_i}\right)$',
              fontsize=13)
ax.set_title('Figure 4 — Normalised DT fusion reactivity', fontsize=12)
ax.set_xticks([4, 5, 6])
ax.set_yticks([-1, 0])
ax.grid(True, which='major', ls='-', color='k', alpha=0.5)
ax.grid(True, which='minor', ls='-', color='k', alpha=0.15)
ax.minorticks_on()

fig.tight_layout()
fig.savefig('fig4_reactivity.png', dpi=150, bbox_inches='tight')
print("Saved fig4_reactivity.png")
plt.show()