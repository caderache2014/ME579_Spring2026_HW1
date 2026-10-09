"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-2(g): Effect of initial conditions.

Five initial conditions spanning a range of prey/predator imbalance are
integrated at baseline parameters and drawn on a single phase portrait.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

ALPHA, BETA, DELTA, GAMMA = 1.0, 0.1, 0.075, 1.5
X_STAR, Y_STAR = GAMMA / DELTA, ALPHA / BETA      # (20, 10)

# Initial conditions chosen to span balanced -> strongly imbalanced.
ICS = [
    ((22.0, 11.0), 'tab:green',  'near equilibrium'),
    ((40.0, 10.0), 'tab:blue',   'prey excess'),
    ((20.0, 25.0), 'tab:red',    'predator excess'),
    ((50.0, 30.0), 'tab:purple', 'both abundant'),
    (( 5.0,  3.0), 'tab:orange', 'both scarce'),
]


def lotka_volterra(t, state):
    x, y = state
    return [ALPHA * x - BETA * x * y,
            DELTA * x * y - GAMMA * y]


def V(x, y):
    return DELTA * x - GAMMA * np.log(x) + BETA * y - ALPHA * np.log(y)


fig, ax = plt.subplots(figsize=(8, 7))
rows = []

for x0, color, label in ICS:
    sol = solve_ivp(lotka_volterra, (0.0, 60.0), list(x0),
                    t_eval=np.linspace(0.0, 60.0, 12000),
                    rtol=1e-11, atol=1e-13)
    x, y = sol.y
    ax.plot(x, y, color=color, lw=1.9, zorder=4, label=f'{label}  {x0}')
    ax.plot(*x0, 'o', color=color, ms=7, mec='k', mew=0.8, zorder=5)
    rows.append((label, x0, V(*x0), x.min(), x.max(), y.min(), y.max()))

# Background vector field.
xg, yg = np.meshgrid(np.linspace(1e-3, 80, 45), np.linspace(1e-3, 46, 45))
ax.streamplot(xg, yg, ALPHA * xg - BETA * xg * yg,
              DELTA * xg * yg - GAMMA * yg,
              density=1.1, color='0.65', linewidth=0.7, arrowsize=0.9)

ax.plot(X_STAR, Y_STAR, 'o', color='k', ms=10, zorder=6)
ax.annotate(r'$(x^*,y^*)=(20,10)$', (X_STAR, Y_STAR),
            textcoords='offset points', xytext=(10, -16), fontsize=11)
ax.plot(0, 0, 's', mfc='white', mec='k', mew=1.8, ms=8, zorder=6)

ax.set_xlim(-2, 80)
ax.set_ylim(-1.2, 46)
ax.set_xlabel('prey $x$')
ax.set_ylabel('predator $y$')
ax.set_title('Lotka-Volterra orbits from five initial conditions\n'
             r'($\alpha=1.0$, $\beta=0.1$, $\delta=0.075$, $\gamma=1.5$)')
ax.legend(loc='upper right', framealpha=0.95)

fig.tight_layout()
fig.savefig('HW1_2g_initial_conditions.png', dpi=200, bbox_inches='tight')

hdr = (f"{'initial condition':>20} {'V':>8} {'prey min':>9} {'prey max':>9} "
       f"{'pred min':>9} {'pred max':>9}")
print(hdr)
print('-' * len(hdr))
for label, x0, v, xmn, xmx, ymn, ymx in rows:
    print(f'{str(x0):>20} {v:>8.3f} {xmn:>9.2f} {xmx:>9.2f} '
          f'{ymn:>9.2f} {ymx:>9.2f}')
print(f'\nV at the fixed point (minimum): {V(X_STAR, Y_STAR):.3f}')

plt.show()