"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(d), step (iii): integrate dx/dt = A x from x0 = [0.2, 0.5]
for t = 0 to 100, with

    A = [[ 0, a],
         [-a, 0]],     a = -0.1

Because A is skew-symmetric the solution is a pure rotation at angular
rate |a|:

    x(t) = R(|a| t) x0,   R = [[cos, -sin],
                               [sin,  cos]]

so the radius |x| is exactly constant. The closed form is overlaid on
the numerical solution as a check.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 11})

# --- System ----------------------------------------------------------
a = -0.1
A = np.array([[0.0, a  ],
              [-a,  0.0]])

X0 = np.array([0.2, 0.5])
T_END = 100.0
OMEGA = abs(a)                      # angular rate, rad per unit time


def rhs(t, x):
    return A @ x


def exact(t, x0=X0):
    c, s = np.cos(OMEGA * t), np.sin(OMEGA * t)
    return c * x0[0] - s * x0[1], s * x0[0] + c * x0[1]


# --- Integrate -------------------------------------------------------
sol = solve_ivp(rhs, (0.0, T_END), list(X0),
                t_eval=np.linspace(0.0, T_END, 5000),
                rtol=1e-11, atol=1e-13)
x1, x2 = sol.y
x1_ex, x2_ex = exact(sol.t)
r = np.hypot(x1, x2)

# --- Plot ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sol.t, x1, color='tab:blue', lw=1.8, label='$x_1(t)$')
ax.plot(sol.t, x2, color='tab:orange', lw=1.8, label='$x_2(t)$')

r0 = np.linalg.norm(X0)
ax.axhline(r0, color='k', ls='--', lw=1.1, label=r'$\pm|x_0|$ (constant)')
ax.axhline(-r0, color='k', ls='--', lw=1.1)

t_chk = np.linspace(0.0, T_END, 25)
ax.plot(t_chk, exact(t_chk)[0], 'k.', ms=5, label='exact (hand)')
ax.plot(t_chk, exact(t_chk)[1], 'k.', ms=5)

# Mark one full period.
T = 2 * np.pi / OMEGA
ax.axvline(T, color='0.6', ls=':', lw=1.2)
ax.annotate(f'one period, $T={T:.1f}$', xy=(T, -0.62),
            xytext=(T + 2, -0.62), fontsize=10, color='#55554f',
            va='center')

ax.axhline(0, color='0.85', lw=0.8, zorder=0)
ax.set_xlim(0, T_END)
ax.set_ylim(-0.72, 0.72)
ax.set_xlabel('time $t$')
ax.set_ylabel('state')
ax.set_title(r'Exercise 1-4(d):  $x_0=[0.2,\,0.5]^T$')
ax.legend(loc='upper right', framealpha=0.95, ncol=2, fontsize=10)
fig.tight_layout()
fig.savefig('HW1_4d_timeseries.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
print(f'x(0)                  : [{x1[0]:.6f}, {x2[0]:.6f}]')
print(f'x(100)                : [{x1[-1]:.6f}, {x2[-1]:.6f}]')
print(f'max |numeric - exact| : '
      f'{max(np.abs(x1-x1_ex).max(), np.abs(x2-x2_ex).max()):.3e}\n')

print(f'period 2*pi/|a|       : {T:.4f}')
print(f'revolutions by t=100  : {T_END/T:.4f}')
print(f'angle swept           : {np.degrees(OMEGA*T_END):.1f} deg\n')

print(f'radius at t=0         : {r[0]:.10f}')
print(f'radius at t=100       : {r[-1]:.10f}')
print(f'max radius drift      : {np.abs(r - r[0]).max():.3e}')
print(f'amplitude of each cpt : {r0:.6f}  (= |x0|)')

plt.show()