"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-2(c): Numerical integration of the Lotka-Volterra system.

    x_dot = alpha*x - beta*x*y      (prey)
    y_dot = delta*x*y - gamma*y     (predator)

Integrated from t = 0 to t = 50 with (x0, y0) = (10, 5).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 11})

# --- Parameters (baseline values from the assignment) ----------------
ALPHA, BETA, DELTA, GAMMA = 1.0, 0.1, 0.075, 1.5


def lotka_volterra(t, state, alpha=ALPHA, beta=BETA, delta=DELTA, gamma=GAMMA):
    """Right-hand side of the Lotka-Volterra system."""
    x, y = state
    return [alpha * x - beta * x * y,
            delta * x * y - gamma * y]


# --- Integrate -------------------------------------------------------
# Tight tolerances matter here: the orbits are closed, so any numerical
# dissipation shows up as a slow drift in amplitude over many cycles.
t_span = (0.0, 50.0)
t_eval = np.linspace(*t_span, 5000)
sol = solve_ivp(lotka_volterra, t_span, [10.0, 5.0], t_eval=t_eval,
                rtol=1e-10, atol=1e-12)

x, y = sol.y

# --- Plot ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sol.t, x, lw=1.8, color='tab:green', label='prey $x(t)$')
ax.plot(sol.t, y, lw=1.8, color='tab:red', label='predator $y(t)$')

x_star, y_star = GAMMA / DELTA, ALPHA / BETA
ax.axhline(x_star, color='tab:green', ls=':', lw=1.2,
           label=rf'$x^*={x_star:.0f}$')
ax.axhline(y_star, color='tab:red', ls=':', lw=1.2,
           label=rf'$y^*={y_star:.0f}$')

ax.set_xlabel('time $t$')
ax.set_ylabel('population')
ax.set_xlim(*t_span)
ax.set_ylim(0, 50)
ax.set_title('Lotka-Volterra time series, '
             r'$(x_0,y_0)=(10,5)$')
ax.legend(loc='upper right', ncol=2, framealpha=0.95)
fig.tight_layout()
fig.savefig('HW1_2c_timeseries.png', dpi=200, bbox_inches='tight')

# --- Numbers to quote in the writeup ---------------------------------
# Peak locations, found from the sampled trajectory.
def peaks(signal, t):
    i = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
    return t[i], signal[i]

tx, vx = peaks(x, sol.t)
ty, vy = peaks(y, sol.t)

print(f'fixed point      : (x*, y*) = ({x_star:.1f}, {y_star:.1f})')
print(f'prey     range   : {x.min():.2f} to {x.max():.2f}')
print(f'predator range   : {y.min():.2f} to {y.max():.2f}')
print(f'period (prey pks): {np.mean(np.diff(tx)):.3f}')
print(f'linear prediction: {2*np.pi/np.sqrt(ALPHA*GAMMA):.3f}')
print(f'predator lag     : {np.mean(ty[:len(tx)] - tx[:len(ty)]):.3f}')
print(f'amplitude drift  : first prey peak {vx[0]:.3f}, last {vx[-1]:.3f}')

plt.show()