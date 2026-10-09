"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(b), step (iv): phase portrait of dx/dt = A x with the
purely real eigenvector overlaid.

    A = [[a, eps],
         [0, a  ]],     a = -0.1,  eps = 0.05

The eigenvalue lambda = a is repeated but the eigenspace is only
one-dimensional (step ii), so there is a SINGLE eigenvector direction
v = [1, 0]. Extra trajectories are drawn to show that every orbit
approaches the origin tangent to that one direction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

# --- System ----------------------------------------------------------
a, eps = -0.1, 0.05
A = np.array([[a,   eps],
              [0.0, a  ]])

X0 = (0.2, 0.5)      # initial condition specified in step (iii)
LIM = 0.6
T_END = 100.0


def rhs(t, x):
    return A @ x


def trajectory(x0, t_end=T_END, n=4000):
    sol = solve_ivp(rhs, (0.0, t_end), list(x0),
                    t_eval=np.linspace(0.0, t_end, n),
                    rtol=1e-10, atol=1e-12)
    return sol.y


# --- Phase portrait --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6.5))

g = np.linspace(-LIM, LIM, 25)
X1, X2 = np.meshgrid(g, g)
ax.streamplot(X1, X2,
              A[0, 0] * X1 + A[0, 1] * X2,
              A[1, 0] * X1 + A[1, 1] * X2,
              density=1.2, color='0.65', linewidth=0.7, arrowsize=0.9)

# The single eigenvector direction. numpy returns two columns but they
# are the same line (see step ii), so only one is drawn.
lam, vec = np.linalg.eig(A)
v = vec[:, 0].real / np.linalg.norm(vec[:, 0].real)
ax.plot([-LIM * v[0], LIM * v[0]], [-LIM * v[1], LIM * v[1]],
        color='tab:green', lw=2.4, zorder=3,
        label=rf'$v=[1,0]^T$,  $\lambda={lam[0].real:+.2f}$')

# Extra trajectories, to show the curvature and common tangency.
for x0 in [(-0.5, 0.45), (0.45, 0.35), (-0.45, -0.35), (0.5, -0.45),
           (0.05, 0.55), (-0.05, -0.55)]:
    y = trajectory(x0)
    ax.plot(y[0], y[1], color='0.35', lw=1.2, zorder=4)

# The specified initial condition.
y = trajectory(X0)
ax.plot(y[0], y[1], color='tab:blue', lw=2.4, zorder=5, label='trajectory')
ax.plot(*X0, 'o', color='tab:blue', ms=8, mec='k', mew=0.8, zorder=6,
        label=r'$x_0=[0.2,\,0.5]^T$')
ax.plot(0, 0, 'o', color='k', ms=9, zorder=6)

ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect('equal')
ax.axhline(0, color='0.85', lw=0.8, zorder=0)
ax.axvline(0, color='0.85', lw=0.8, zorder=0)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Exercise 1-4(b): phase portrait')
ax.legend(loc='lower right', framealpha=0.95)

fig.tight_layout()
fig.savefig('HW1_4b_phase.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
print(f'eigenvalues       : {lam}')
print(f'eigenvector       : {v}')
print('\nangle of the state vector from the x1 axis, along the trajectory:')
for t_mark in [0, 5, 20, 50, 100]:
    yy = trajectory(X0, t_end=max(t_mark, 1e-6), n=2)
    ang = np.degrees(np.arctan2(yy[1][-1], yy[0][-1]))
    print(f'  t = {t_mark:>4}: {ang:7.3f} deg')

plt.show()