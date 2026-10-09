"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(d), step (iv): phase portrait of dx/dt = A x.

    A = [[ 0, a],
         [-a, 0]],     a = -0.1

The eigenvalues are purely imaginary, so the eigenvectors are complex
and define no direction in the real x1-x2 plane. NO eigenvectors are
plotted -- the exercise asks for them only for purely real eigenvectors.

Because A is skew-symmetric the radius is exactly conserved, so every
orbit is a closed circle about the origin.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

# --- System ----------------------------------------------------------
a = -0.1
A = np.array([[0.0, a  ],
              [-a,  0.0]])

X0 = (0.2, 0.5)
LIM = 0.6
PERIOD = 2 * np.pi / abs(a)


def rhs(t, x):
    return A @ x


def trajectory(x0, t_end, n=4000):
    sol = solve_ivp(rhs, (0.0, t_end), list(x0),
                    t_eval=np.linspace(0.0, t_end, n),
                    rtol=1e-11, atol=1e-13)
    return sol.y


# --- Phase portrait --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6.5))

g = np.linspace(-LIM, LIM, 25)
X1, X2 = np.meshgrid(g, g)
ax.streamplot(X1, X2,
              A[0, 0] * X1 + A[0, 1] * X2,
              A[1, 0] * X1 + A[1, 1] * X2,
              density=1.2, color='0.65', linewidth=0.7, arrowsize=0.9)

# A family of concentric closed orbits, each run for exactly one period.
for r in [0.15, 0.30, 0.45]:
    y = trajectory((r, 0.0), PERIOD)
    ax.plot(y[0], y[1], color='0.35', lw=1.2, zorder=4)

# The specified initial condition, one full period.
y = trajectory(X0, PERIOD)
ax.plot(y[0], y[1], color='tab:blue', lw=2.4, zorder=5,
        label='trajectory (one period)')
ax.plot(*X0, 'o', color='tab:blue', ms=8, mec='k', mew=0.8, zorder=6,
        label=r'$x_0=[0.2,\,0.5]^T$')
ax.plot(0, 0, 'o', color='k', ms=9, zorder=6, label='center')

ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_aspect('equal')
ax.axhline(0, color='0.85', lw=0.8, zorder=0)
ax.axvline(0, color='0.85', lw=0.8, zorder=0)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Exercise 1-4(d): phase portrait')
ax.legend(loc='lower right', framealpha=0.95)

fig.tight_layout()
fig.savefig('HW1_4d_phase.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
lam = np.linalg.eigvals(A)
r_traj = np.hypot(y[0], y[1])
closure = np.linalg.norm(y[:, -1] - np.array(X0))

print(f'eigenvalues              : {lam}')
print('eigenvectors are complex -> not plotted\n')
print(f'period                   : {PERIOD:.4f}')
print(f'radius (min, max)        : {r_traj.min():.10f}, {r_traj.max():.10f}')
print(f'|x(T) - x(0)|            : {closure:.3e}   (orbit closes)')
print(f'rotation sense           : '
      f'{"counter-clockwise" if (A @ np.array([1.0, 0.0]))[1] > 0 else "clockwise"}')
print(f'det(A)                   : {np.linalg.det(A):+.4f}  '
      f'(nonzero -> origin is the only fixed point)')

plt.show()