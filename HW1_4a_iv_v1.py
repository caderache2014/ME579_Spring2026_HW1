"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(a), step (iv): phase portrait of dx/dt = A x with the
purely real eigenvectors overlaid.

    A = [[a, 0],
         [0, a]],     a = -0.1

This script produces the phase portrait only. The eigenvalues and
eigenvectors of steps (i)-(ii) and the time series of step (iii) are in
their own scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

# --- System ----------------------------------------------------------
a = -0.1
A = np.array([[a,   0.0],
              [0.0, a  ]])

X0 = (0.2, 0.5)      # initial condition specified in step (iii)
LIM = 0.6            # axis half-width
T_END = 100.0


def rhs(t, x):
    return A @ x


def eigen_real(A):
    """Unit eigenvectors of A, and whether they are purely real."""
    lam, vec = np.linalg.eig(A)
    vec = vec / np.linalg.norm(vec, axis=0)
    return lam, vec, np.allclose(vec.imag, 0.0)


# --- Phase portrait --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6.5))

# Vector field.
g = np.linspace(-LIM, LIM, 25)
X1, X2 = np.meshgrid(g, g)
dX1 = A[0, 0] * X1 + A[0, 1] * X2
dX2 = A[1, 0] * X1 + A[1, 1] * X2
ax.streamplot(X1, X2, dX1, dX2, density=1.2, color='0.65',
              linewidth=0.7, arrowsize=0.9)

# A = a*I means (A - lambda*I) is the zero matrix, so EVERY direction is
# an eigendirection. The faint radial lines show the full 2-D eigenspace,
# making clear that the two bold eigenvectors below are an arbitrary
# choice of basis rather than a pair of special axes.
for ang in np.arange(0, 180, 15):
    d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    ax.plot([-LIM * d[0], LIM * d[0]], [-LIM * d[1], LIM * d[1]],
            color='tab:green', lw=0.8, alpha=0.35, zorder=2)

# Eigenvectors (plotted because the eigenvalues here are purely real).
lam, vec, purely_real = eigen_real(A)
if purely_real:
    for k, color in enumerate(['tab:green', 'tab:purple']):
        v = vec[:, k].real
        ax.plot([-LIM * v[0], LIM * v[0]], [-LIM * v[1], LIM * v[1]],
                color=color, lw=2.2, zorder=3,
                label=rf'$v_{k+1}$,  $\lambda={lam[k].real:+.2f}$')

# Trajectory from the specified initial condition.
sol = solve_ivp(rhs, (0.0, T_END), list(X0),
                t_eval=np.linspace(0.0, T_END, 4000),
                rtol=1e-10, atol=1e-12)
ax.plot(sol.y[0], sol.y[1], color='tab:blue', lw=2.4, zorder=5,
        label='trajectory')
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
ax.set_title('Exercise 1-4(a): phase portrait')
ax.legend(loc='lower right', framealpha=0.95)

fig.tight_layout()
fig.savefig('HW1_4a_phase.png', dpi=200, bbox_inches='tight')

print(f'eigenvalues  : {lam}')
print(f'eigenvectors :\n{vec.real}')
print(f'x(0)         : [{sol.y[0][0]:.4f}, {sol.y[1][0]:.4f}]')
print(f'x(T)         : [{sol.y[0][-1]:.3e}, {sol.y[1][-1]:.3e}]')
print(f'x2/x1 held at: {sol.y[1][-1]/sol.y[0][-1]:.6f}  '
      f'(initial ratio {X0[1]/X0[0]:.6f})')

plt.show()