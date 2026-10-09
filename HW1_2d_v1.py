"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-2(d): Parameter sweep over the prey growth rate alpha.

    x_dot = alpha*x - beta*x*y      (prey)
    y_dot = delta*x*y - gamma*y     (predator)

beta, delta, gamma are held at baseline; alpha takes the values
0.5, 1.0, 2.0, 3.0.

SEEDING CONVENTION
------------------
The non-trivial fixed point moves with alpha, since y* = alpha/beta.
Starting every panel from the same absolute coordinates would therefore
apply a different-sized disturbance in each panel, confounding the effect
of alpha with the effect of the initial condition.

Instead each orbit is seeded at (r * x*, y*): predators start AT their
equilibrium value, prey start at a fraction r of theirs. Every panel thus
receives the same proportional disturbance.

Two properties make this line a natural choice:
  1. Prey extrema occur where x_dot = x(alpha - beta*y) = 0, i.e. exactly
     at y = y*. The seed is therefore a turning point of its own orbit,
     so x0 = x_min by construction.
  2. x* = gamma/delta does not depend on alpha, so r*x* is the same
     ABSOLUTE prey value in all four panels as well as the same relative
     one. The seeding is defensible under either convention.

Three nested values of r span the near-linear to strongly nonlinear
regime, so the transition from ellipse to distorted teardrop is visible.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 14,
                     'axes.labelsize': 13, 'legend.fontsize': 12})

# --- Fixed parameters ------------------------------------------------
BETA, DELTA, GAMMA = 0.1, 0.075, 1.5
ALPHAS = [0.5, 1.0, 2.0, 3.0]

# Seeding ratios r, and the style each is drawn in, outermost orbit
# first. r = 0.50 is drawn boldest because it is the orbit used for the
# quantitative measurements printed at the end.
ORBITS = [
    (0.50, 'tab:blue',  2.2, r'$r=0.50$  (largest disturbance)'),
    (0.70, 'tab:cyan',  1.6, r'$r=0.70$'),
    (0.85, 'tab:olive', 1.6, r'$r=0.85$  (smallest disturbance)'),
]


def lotka_volterra(t, state, alpha, beta=BETA, delta=DELTA, gamma=GAMMA):
    x, y = state
    return [alpha * x - beta * x * y,
            delta * x * y - gamma * y]


def fixed_point(alpha):
    """Non-trivial fixed point (gamma/delta, alpha/beta)."""
    return GAMMA / DELTA, alpha / BETA


def integrate(alpha, x0, t_end, n=8000):
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(lotka_volterra, (0.0, t_end), x0, args=(alpha,),
                    t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1]


def measure_period(t, signal):
    """Mean peak-to-peak spacing of a sampled periodic signal."""
    i = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
    return np.mean(np.diff(t[i])) if len(i) > 1 else np.nan


# --- Build the figure ------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
results = []

for ax, alpha in zip(axes.flat, ALPHAS):
    x_star, y_star = fixed_point(alpha)

    # Integrate for roughly eight small-oscillation periods so the
    # period estimate averages over several complete cycles.
    t_end = 8 * 2 * np.pi / np.sqrt(alpha * GAMMA)
    trajectories = [integrate(alpha, [r * x_star, y_star], t_end)
                    for r, _, _, _ in ORBITS]

    # The outermost orbit sets the plot window.
    x_hi = 1.12 * max(tr[1].max() for tr in trajectories)
    y_hi = 1.12 * max(tr[2].max() for tr in trajectories)

    # Background vector field.
    xg, yg = np.meshgrid(np.linspace(1e-3, x_hi, 40),
                         np.linspace(1e-3, y_hi, 40))
    dx = alpha * xg - BETA * xg * yg
    dy = DELTA * xg * yg - GAMMA * yg
    ax.streamplot(xg, yg, dx, dy, density=1.1, color='0.6',
                  linewidth=0.7, arrowsize=0.9)

    # Closed orbits, one per seeding ratio, with their seed points marked.
    for (t, x, y), (r, color, lw, _) in zip(trajectories, ORBITS):
        ax.plot(x, y, color=color, lw=lw, zorder=4)
        ax.plot(r * x_star, y_star, 'o', color=color, ms=5,
                mec='k', mew=0.6, zorder=5)

    # Fixed points.
    ax.plot(x_star, y_star, 'o', color='k', ms=9, zorder=6)
    ax.plot(0, 0, 's', mfc='white', mec='k', mew=1.8, ms=8, zorder=6)

    # Dotted guides through the coexistence fixed point.
    ax.axhline(y_star, color='k', ls=':', lw=0.9, alpha=0.5, zorder=1)
    ax.axvline(x_star, color='k', ls=':', lw=0.9, alpha=0.5, zorder=1)

    ax.set_xlim(-0.025 * x_hi, x_hi)
    ax.set_ylim(-0.025 * y_hi, y_hi)
    ax.set_xlabel('prey $x$')
    ax.set_ylabel('predator $y$')
    ax.set_title(rf'$\alpha={alpha}$:   $(x^*,y^*)=({x_star:.0f},\,{y_star:.0f})$')

    # --- Diagnostics for the bold (r = 0.50) orbit -------------------
    t, x, y = trajectories[0]
    results.append(dict(
        alpha=alpha, x_star=x_star, y_star=y_star,
        x_min=x.min(), x_max=x.max(), y_min=y.min(), y_max=y.max(),
        T_num=measure_period(t, x),
        T_lin=2 * np.pi / np.sqrt(alpha * GAMMA)))

# --- One shared legend for all four panels ---------------------------
handles = [Line2D([], [], color=c, lw=lw, label=lab)
           for _, c, lw, lab in ORBITS]
handles += [
    Line2D([], [], color='k', marker='o', ls='none', ms=9,
           label=r'coexistence fixed point $(x^*,y^*)$'),
    Line2D([], [], color='k', marker='s', ls='none', ms=8,
           mfc='white', mew=1.8, label=r'extinction fixed point $(0,0)$'),
    Line2D([], [], color='0.6', lw=0.9, label='vector field'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.005))

fig.suptitle(r'Lotka-Volterra phase portraits, sweep over $\alpha$   '
             r'($\beta=0.1$, $\delta=0.075$, $\gamma=1.5$).'
             '\n'
             r'Each orbit is seeded at $(r\,x^*,\,y^*)$: the same relative '
             r'disturbance in every panel.', fontsize=13)
fig.tight_layout(rect=[0, 0.085, 1, 1])
fig.savefig('HW1_2d_alpha_sweep.png', dpi=200, bbox_inches='tight')

# --- Table for the writeup -------------------------------------------
hdr = (f"{'alpha':>6} {'x*':>5} {'y*':>5} {'prey range':>20} "
       f"{'predator range':>22} {'T_num':>8} {'T_lin':>8}")
print('Measurements for the bold r = 0.50 orbit:\n')
print(hdr)
print('-' * len(hdr))
for r in results:
    print(f"{r['alpha']:>6.1f} {r['x_star']:>5.0f} {r['y_star']:>5.0f} "
          f"{r['x_min']:>9.2f} to {r['x_max']:>7.2f} "
          f"{r['y_min']:>10.2f} to {r['y_max']:>7.2f} "
          f"{r['T_num']:>8.3f} {r['T_lin']:>8.3f}")

plt.show()