"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(c), step (iii): integrate dx/dt = A x from x0 = [0.2, 0.5]
for t = 0 to 100, with

    A = [[ a,   eps],
         [-eps, a  ]],     a = -0.1,  eps = 0.05

The closed form derived in step (ii),

    x(t) = exp(a t) * R(eps t) * x0,   R = [[ cos, sin],
                                            [-sin, cos]]

is overlaid to verify the numerical integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 11})

# --- System ----------------------------------------------------------
a, eps = -0.1, 0.05
A = np.array([[a,    eps],
              [-eps, a  ]])

X0 = np.array([0.2, 0.5])
T_END = 100.0


def rhs(t, x):
    return A @ x


def exact(t, x0=X0):
    """Rotation by eps*t, scaled by exp(a*t)."""
    c, s = np.cos(eps * t), np.sin(eps * t)
    x1 = np.exp(a * t) * (c * x0[0] + s * x0[1])
    x2 = np.exp(a * t) * (-s * x0[0] + c * x0[1])
    return x1, x2


# --- Integrate -------------------------------------------------------
sol = solve_ivp(rhs, (0.0, T_END), list(X0),
                t_eval=np.linspace(0.0, T_END, 4000),
                rtol=1e-10, atol=1e-12)
x1, x2 = sol.y
x1_ex, x2_ex = exact(sol.t)

# --- Plot ------------------------------------------------------------
fig, (ax, ax_n) = plt.subplots(1, 2, figsize=(12, 4.5))
ax.plot(sol.t, x1, color='tab:blue', lw=1.8, label='$x_1(t)$')
ax.plot(sol.t, x2, color='tab:orange', lw=1.8, label='$x_2(t)$')

# Decay envelope: radius r(t) = |x0| exp(a t).
r0 = np.linalg.norm(X0)
env = r0 * np.exp(a * sol.t)
ax.plot(sol.t, env, 'k--', lw=1.1, label=r'$\pm|x_0|e^{at}$ envelope')
ax.plot(sol.t, -env, 'k--', lw=1.1)

t_chk = np.linspace(0.0, T_END, 25)
ax.plot(t_chk, exact(t_chk)[0], 'k.', ms=5, label='exact (hand)')
ax.plot(t_chk, exact(t_chk)[1], 'k.', ms=5)

ax.axhline(0, color='0.85', lw=0.8, zorder=0)
ax.set_xlim(0, T_END)
ax.set_xlabel('time $t$')
ax.set_ylabel('state')
ax.set_title(r'Response,  $x_0=[0.2,\,0.5]^T$')
ax.legend(loc='upper right', framealpha=0.95, fontsize=10)

# Right panel: divide out exp(a t) so only the rotation remains. This
# makes the oscillation visible, which the raw response hides because the
# decay time constant (10) is far shorter than the period (125.7).
decay = np.exp(a * sol.t)
ax_n.plot(sol.t, x1 / decay, color='tab:blue', lw=1.8,
          label='$x_1(t)\\,e^{-at}$')
ax_n.plot(sol.t, x2 / decay, color='tab:orange', lw=1.8,
          label='$x_2(t)\\,e^{-at}$')
ax_n.axhline(0, color='0.85', lw=0.8, zorder=0)
ax_n.set_xlim(0, T_END)
ax_n.set_xlabel('time $t$')
ax_n.set_ylabel('state / decay envelope')
ax_n.set_title('Decay envelope removed: the underlying rotation')
ax_n.legend(loc='lower left', framealpha=0.95, fontsize=10)

fig.tight_layout()
fig.savefig('HW1_4c_timeseries.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
r = np.hypot(x1, x2)
print(f'x(0)                  : [{x1[0]:.6f}, {x2[0]:.6f}]')
print(f'x(100)                : [{x1[-1]:.4e}, {x2[-1]:.4e}]')
print(f'max |numeric - exact| : {max(np.abs(x1-x1_ex).max(), np.abs(x2-x2_ex).max()):.3e}\n')

print(f'oscillation period    : {2*np.pi/eps:.2f}')
print(f'angle swept by t=100  : {np.degrees(eps*T_END):.1f} deg '
      f'({eps*T_END/(2*np.pi):.3f} of a revolution)\n')

print(f'radius at t=0         : {r[0]:.6f}')
print(f'radius at t=100       : {r[-1]:.4e}')
print(f'ratio r(100)/r(0)     : {r[-1]/r[0]:.4e}   (exp(-10) = '
      f'{np.exp(-10):.4e})\n')

# x1 rises before decaying: the rotation feeds x2 into x1 early on.
i_pk = np.argmax(x1)
print(f'x1 peaks at t = {sol.t[i_pk]:.2f}, value {x1[i_pk]:.6f} '
      f'(started at {X0[0]:.3f})')
i_z = np.where(np.diff(np.sign(x2)))[0]
print(f'x2 first crosses zero : t = {sol.t[i_z[0]]:.2f}'
      if len(i_z) else 'x2 does not cross zero')

plt.show()