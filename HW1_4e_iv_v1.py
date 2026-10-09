"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(e), step (iv): phase portrait of dx/dt = A x with the
purely real eigenvectors overlaid.

    A = [[a,  0],
         [0, -a]],     a = -0.1

Eigenvalues are real and of opposite sign, so BOTH eigenvectors are
real and are plotted:

    v1 = [1, 0],  lambda = -0.1   stable   (trajectories come in)
    v2 = [0, 1],  lambda = +0.1   unstable (trajectories go out)

Trajectories are hyperbolas x1*x2 = const, since tr(A) = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

# --- System ----------------------------------------------------------
a = -0.1
A = np.array([[a,   0.0],
              [0.0, -a ]])

X0 = (0.2, 0.5)
LIM = 0.6


def rhs(t, x):
    return A @ x


def trajectory(x0, t_end, n=4000):
    sol = solve_ivp(rhs, (0.0, t_end), list(x0),
                    t_eval=np.linspace(0.0, t_end, n),
                    rtol=1e-11, atol=1e-14)
    return sol.y


# --- Phase portrait --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6.5))

g = np.linspace(-LIM, LIM, 25)
X1, X2 = np.meshgrid(g, g)
ax.streamplot(X1, X2,
              A[0, 0] * X1 + A[0, 1] * X2,
              A[1, 0] * X1 + A[1, 1] * X2,
              density=1.2, color='0.65', linewidth=0.7, arrowsize=0.9)

# Eigenvector lines, with arrows showing the direction of motion.
lam, vec = np.linalg.eig(A)
colors = {True: 'tab:green', False: 'tab:red'}     # True = stable
for k in range(2):
    v = vec[:, k].real / np.linalg.norm(vec[:, k].real)
    stable = lam[k].real < 0
    lab = (rf'$v_{k+1}=[{v[0]:.0f},{v[1]:.0f}]^T$, '
           rf'$\lambda={lam[k].real:+.1f}$ '
           + ('(stable)' if stable else '(unstable)'))
    ax.plot([-LIM * v[0], LIM * v[0]], [-LIM * v[1], LIM * v[1]],
            color=colors[stable], lw=2.4, zorder=3, label=lab)
    # arrows: inward on the stable axis, outward on the unstable one
    for sgn in (+1, -1):
        tail = 0.30 if stable else 0.18
        head = 0.18 if stable else 0.30
        ax.annotate('', xy=(sgn * head * v[0], sgn * head * v[1]),
                    xytext=(sgn * tail * v[0], sgn * tail * v[1]),
                    arrowprops=dict(arrowstyle='-|>', lw=2.0,
                                    color=colors[stable]), zorder=7)

# Hyperbolic trajectories in all four quadrants.
for x0 in [(0.5, 0.03), (-0.5, 0.03), (0.5, -0.03), (-0.5, -0.03),
           (0.5, 0.12), (-0.5, -0.12)]:
    y = trajectory(x0, 40.0)
    ax.plot(y[0], y[1], color='0.35', lw=1.2, zorder=4)

# The specified initial condition.
y = trajectory(X0, 11.0)
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
ax.set_title('Exercise 1-4(e): phase portrait')
ax.legend(loc='lower right', framealpha=0.95, fontsize=9)

fig.tight_layout()
fig.savefig('HW1_4e_phase.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
print(f'eigenvalues  : {lam}')
print(f'eigenvectors :\n{vec.real}\n')
print('Starting ON the stable axis decays to the origin:')
ys = trajectory((0.4, 0.0), 100.0)
print(f'  x0 = [0.4, 0]  ->  x(100) = [{ys[0][-1]:.4e}, {ys[1][-1]:.4e}]\n')
print('Starting just OFF it diverges:')
yo = trajectory((0.4, 1e-6), 100.0)
print(f'  x0 = [0.4, 1e-6] -> x(100) = [{yo[0][-1]:.4e}, {yo[1][-1]:.4e}]')
print('\n-> the stable axis is a measure-zero set; any x2 != 0 grows.')

plt.show()