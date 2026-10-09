"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(b), step (iii): integrate dx/dt = A x from x0 = [0.2, 0.5]
for t = 0 to 100, with

    A = [[a, eps],
         [0, a  ]],     a = -0.1,  eps = 0.05

The closed-form solution derived by hand in step (ii),

    x1(t) = (x1(0) + eps*x2(0)*t) * exp(a*t)
    x2(t) =  x2(0)               * exp(a*t)

is overlaid as a dashed line to verify the numerical integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 11})

# --- System ----------------------------------------------------------
a, eps = -0.1, 0.05
A = np.array([[a,   eps],
              [0.0, a  ]])

X0 = (0.2, 0.5)
T_END = 100.0


def rhs(t, x):
    return A @ x


def exact(t, x0=X0):
    """Hand-derived solution from step (ii)."""
    x1 = (x0[0] + eps * x0[1] * t) * np.exp(a * t)
    x2 = x0[1] * np.exp(a * t)
    return x1, x2


# --- Integrate -------------------------------------------------------
t_eval = np.linspace(0.0, T_END, 4000)
sol = solve_ivp(rhs, (0.0, T_END), list(X0), t_eval=t_eval,
                rtol=1e-10, atol=1e-12)
x1, x2 = sol.y
x1_ex, x2_ex = exact(sol.t)

# --- Plot ------------------------------------------------------------
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5))

for ax in (ax_full, ax_zoom):
    ax.plot(sol.t, x1, color='tab:blue', lw=1.8, label='$x_1(t)$')
    ax.plot(sol.t, x2, color='tab:orange', lw=1.8, label='$x_2(t)$')
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('state')

# Verification markers on the full view only.
t_chk = np.linspace(0.0, T_END, 25)
ax_full.plot(t_chk, exact(t_chk)[0], 'k.', ms=5, label='exact (hand)')
ax_full.plot(t_chk, exact(t_chk)[1], 'k.', ms=5)
ax_full.set_xlim(0, T_END)
ax_full.set_title('Full response')
ax_full.legend(loc='upper right', framealpha=0.95)

# Zoom on the early transient, where x1 rises before decaying.
t_peak = -1.0 / a - X0[0] / (eps * X0[1])          # d(x1)/dt = 0
ax_zoom.axvline(t_peak, color='0.5', ls='--', lw=1.2)
ax_zoom.annotate(f'$x_1$ peaks at $t={t_peak:.1f}$',
                 (t_peak, exact(t_peak)[0]),
                 textcoords='offset points', xytext=(10, 14), fontsize=11)
ax_zoom.set_xlim(0, 12)
ax_zoom.set_ylim(0.15, 0.55)
ax_zoom.set_title('Zoom: early transient')

fig.tight_layout()
fig.savefig('HW1_4b_timeseries.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
print(f'x(0)                  : [{x1[0]:.6f}, {x2[0]:.6f}]')
print(f'x(100)                : [{x1[-1]:.4e}, {x2[-1]:.4e}]')
print(f'max |numeric - exact| : {max(np.abs(x1-x1_ex).max(), np.abs(x2-x2_ex).max()):.3e}')
print()
print(f'analytic peak time    : t = {t_peak:.4f}')
print(f'x1 at t = 0           : {X0[0]:.6f}')
print(f'x1 at peak            : {exact(t_peak)[0]:.6f}')
print(f'overshoot             : {100*(exact(t_peak)[0]/X0[0] - 1):.2f} %')
print()
print(f'ratio x2/x1 at t = 0  : {x2[0]/x1[0]:.4f}')
print(f'ratio x2/x1 at t = 100: {x2[-1]/x1[-1]:.4f}')

plt.show()