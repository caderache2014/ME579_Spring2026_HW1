"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-2(f)(ii): Numerical check of the conserved quantity

    V(x, y) = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)

Left panel : V(t) along the four r = 0.50 orbits from part (d).
Right panel: drift |V(t) - V(0)| for the alpha = 1.0 orbit at three
             integrator tolerances, showing that the small residual
             variation is numerical, not physical.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

BETA, DELTA, GAMMA = 0.1, 0.075, 1.5
ALPHAS = [0.5, 1.0, 2.0, 3.0]
R_SEED = 0.50


def lotka_volterra(t, state, alpha):
    x, y = state
    return [alpha * x - BETA * x * y,
            DELTA * x * y - GAMMA * y]


def V(x, y, alpha):
    """Conserved quantity (first integral) of the Lotka-Volterra system."""
    return DELTA * x - GAMMA * np.log(x) + BETA * y - alpha * np.log(y)


def run(alpha, t_end=50.0, rtol=1e-10, atol=1e-12, n=5000):
    x0 = [R_SEED * GAMMA / DELTA, alpha / BETA]
    sol = solve_ivp(lotka_volterra, (0.0, t_end), x0, args=(alpha,),
                    t_eval=np.linspace(0.0, t_end, n), rtol=rtol, atol=atol)
    return sol.t, sol.y[0], sol.y[1]


fig, (ax_v, ax_drift) = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Left: V(t) for each alpha ---------------------------------------
print(f"{'alpha':>6} {'V(0)':>10} {'max|V-V0|':>12} {'relative':>11}")
print('-' * 42)

for alpha, color in zip(ALPHAS, ['tab:blue', 'tab:orange',
                                 'tab:green', 'tab:red']):
    t, x, y = run(alpha)
    v = V(x, y, alpha)
    ax_v.plot(t, v, color=color, lw=1.8, label=rf'$\alpha={alpha}$')
    drift = np.abs(v - v[0]).max()
    print(f'{alpha:>6.1f} {v[0]:>10.6f} {drift:>12.3e} '
          f'{drift/abs(v[0]):>11.3e}')

ax_v.set_xlabel('time $t$')
ax_v.set_ylabel('$V(x(t),\\,y(t))$')
ax_v.set_title('Conserved quantity along each orbit')
ax_v.legend(loc='center right', framealpha=0.95)
ax_v.set_xlim(0, 50)

# --- Right: drift vs integrator tolerance ----------------------------
# 'default' is solve_ivp's built-in rtol=1e-3, atol=1e-6.
tol_cases = [('default (rtol=$10^{-3}$)', 1e-3, 1e-6, 'tab:red'),
             (r'rtol=$10^{-8}$', 1e-8, 1e-10, 'tab:orange'),
             (r'rtol=$10^{-12}$', 1e-12, 1e-14, 'tab:blue')]

for label, rtol, atol, color in tol_cases:
    t, x, y = run(1.0, rtol=rtol, atol=atol)
    v = V(x, y, 1.0)
    ax_drift.semilogy(t, np.abs(v - v[0]) + 1e-18, color=color,
                      lw=1.6, label=label)

ax_drift.set_xlabel('time $t$')
ax_drift.set_ylabel('$|V(t) - V(0)|$')
ax_drift.set_title(r'Drift in $V$ vs. integrator tolerance ($\alpha=1.0$)')
ax_drift.legend(loc='lower right', framealpha=0.95)
ax_drift.set_xlim(0, 50)

fig.tight_layout()
fig.savefig('HW1_2f_conserved_quantity.png', dpi=200, bbox_inches='tight')
plt.show()