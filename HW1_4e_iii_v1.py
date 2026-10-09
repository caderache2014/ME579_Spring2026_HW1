"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(e), step (iii): integrate dx/dt = A x from x0 = [0.2, 0.5]
for t = 0 to 100, with

    A = [[a,  0],
         [0, -a]],     a = -0.1

The matrix is diagonal, so the system decouples completely:

    x1(t) = x1(0) * exp(a t)    (decays)
    x2(t) = x2(0) * exp(-a t)   (grows)

Because the trace is zero the PRODUCT x1*x2 is conserved, so the orbits
are hyperbolas. Both facts are checked below.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 11})

# --- System ----------------------------------------------------------
a = -0.1
A = np.array([[a,   0.0],
              [0.0, -a ]])

X0 = np.array([0.2, 0.5])
T_END = 100.0


def rhs(t, x):
    return A @ x


def exact(t, x0=X0):
    return x0[0] * np.exp(a * t), x0[1] * np.exp(-a * t)


# --- Integrate -------------------------------------------------------
sol = solve_ivp(rhs, (0.0, T_END), list(X0),
                t_eval=np.linspace(0.0, T_END, 4000),
                rtol=1e-11, atol=1e-14)
x1, x2 = sol.y
x1_ex, x2_ex = exact(sol.t)
prod = x1 * x2

# --- Plot ------------------------------------------------------------
fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5))

ax.semilogy(sol.t, x1, color='tab:blue', lw=1.8,
            label='$x_1(t)$  (stable direction)')
ax.semilogy(sol.t, x2, color='tab:orange', lw=1.8,
            label='$x_2(t)$  (unstable direction)')
t_chk = np.linspace(0.0, T_END, 25)
ax.semilogy(t_chk, exact(t_chk)[0], 'k.', ms=5, label='exact (hand)')
ax.semilogy(t_chk, exact(t_chk)[1], 'k.', ms=5)
ax.axhline(1.0, color='0.8', lw=0.8, zorder=0)
ax.set_xlim(0, T_END)
ax.set_xlabel('time $t$')
ax.set_ylabel('state (log scale)')
ax.set_title('Decoupled growth and decay')
ax.legend(loc='upper left', framealpha=0.95, fontsize=10)

ax_p.plot(sol.t, prod, color='tab:green', lw=2.0)
ax_p.axhline(X0[0] * X0[1], color='k', ls='--', lw=1.1,
             label=rf'$x_1(0)\,x_2(0) = {X0[0]*X0[1]:.2f}$')
ax_p.set_xlim(0, T_END)
ax_p.set_ylim(0.0, 0.2)
ax_p.set_xlabel('time $t$')
ax_p.set_ylabel('$x_1(t)\\,x_2(t)$')
ax_p.set_title('The product is conserved')
ax_p.legend(loc='upper right', framealpha=0.95)
ax_p.text(50, 0.045, f'maximum drift over the run: '
          f'{np.abs(prod - prod[0]).max():.1e}',
          ha='center', fontsize=10, color='#55554f')

fig.tight_layout()
fig.savefig('HW1_4e_timeseries.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
print(f'x(0)                  : [{x1[0]:.6f}, {x2[0]:.6f}]')
print(f'x(100)                : [{x1[-1]:.4e}, {x2[-1]:.4e}]')
print(f'max |numeric - exact| : '
      f'{max(np.abs(x1-x1_ex).max(), np.abs(x2/x2_ex - 1).max()):.3e}\n')

print(f'x1 ratio over run     : {x1[-1]/x1[0]:.4e}   '
      f'(exp(-10) = {np.exp(-10):.4e})')
print(f'x2 ratio over run     : {x2[-1]/x2[0]:.4e}   '
      f'(exp(+10) = {np.exp(10):.4e})\n')

print(f'product at t=0        : {prod[0]:.10f}')
print(f'product at t=100      : {prod[-1]:.10f}')
print(f'max product drift     : {np.abs(prod - prod[0]).max():.3e}')
print(f'|x| at t=0, t=100     : {np.hypot(x1[0], x2[0]):.4f}, '
      f'{np.hypot(x1[-1], x2[-1]):.4e}')

plt.show()