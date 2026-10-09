"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(c), step (iv): phase portrait of dx/dt = A x.

    A = [[ a,   eps],
         [-eps, a  ]],     a = -0.1,  eps = 0.05

The eigenvalues are complex (lambda = a +/- i*eps), so the eigenvectors
are complex as well. They do not correspond to directions in the real
x1-x2 plane, so NO eigenvectors are plotted here -- the exercise asks
for them only "for purely real eigenvectors".

Dashed circles are drawn instead: the radius obeys r(t) = |x0| exp(a t)
independently of angle, which is what makes this a circular spiral
rather than an elliptical one.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})

# --- System ----------------------------------------------------------
a, eps = -0.1, 0.05
A = np.array([[a,    eps],
              [-eps, a  ]])

X0 = (0.2, 0.5)
LIM = 0.6
T_END = 100.0


def rhs(t, x):
    return A @ x


def trajectory(x0, t_end=T_END, n=6000):
    sol = solve_ivp(rhs, (0.0, t_end), list(x0),
                    t_eval=np.linspace(0.0, t_end, n),
                    rtol=1e-11, atol=1e-13)
    return sol.t, sol.y


# --- Phase portrait --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6.5))

g = np.linspace(-LIM, LIM, 25)
X1, X2 = np.meshgrid(g, g)
ax.streamplot(X1, X2,
              A[0, 0] * X1 + A[0, 1] * X2,
              A[1, 0] * X1 + A[1, 1] * X2,
              density=1.2, color='0.65', linewidth=0.7, arrowsize=0.9)

# Decay envelope: circles of radius |x0|*exp(a*t) at selected times.
r0 = np.linalg.norm(X0)
th = np.linspace(0, 2 * np.pi, 400)
for t_mark in [0.0, 20.0, 40.0]:
    r = r0 * np.exp(a * t_mark)
    ax.plot(r * np.cos(th), r * np.sin(th), ls='--', lw=1.0,
            color='tab:green', alpha=0.8, zorder=3)
    ax.annotate(f'$t={t_mark:.0f}$', xy=(0, r), fontsize=8.5,
                color='tab:green', ha='center', va='bottom', zorder=4)
ax.plot([], [], ls='--', lw=1.0, color='tab:green',
        label=r'$r=|x_0|e^{at}$ at $t=0,20,40$')

# Secondary trajectories.
for x0 in [(-0.45, 0.2), (0.3, -0.45), (-0.25, -0.4)]:
    _, y = trajectory(x0)
    ax.plot(y[0], y[1], color='0.35', lw=1.2, zorder=4)

# The specified initial condition.
t, y = trajectory(X0)
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
ax.set_title('Exercise 1-4(c): phase portrait')
ax.legend(loc='lower right', framealpha=0.95)

fig.tight_layout()
fig.savefig('HW1_4c_phase.png', dpi=200, bbox_inches='tight')

# --- Numbers for the writeup -----------------------------------------
lam = np.linalg.eigvals(A)
r = np.hypot(y[0], y[1])
ang = np.degrees(np.unwrap(np.arctan2(y[1], y[0])))

print(f'eigenvalues           : {lam}')
print('eigenvectors are complex -> not plotted\n')
print(f'radius at t=0         : {r[0]:.6f}')
print(f'radius at t=100       : {r[-1]:.4e}')
print(f'r(100)/r(0)           : {r[-1]/r[0]:.4e}   '
      f'(exp(-10) = {np.exp(-10):.4e})\n')
print(f'angle swept over run  : {ang[0]-ang[-1]:.1f} deg '
      f'({(ang[0]-ang[-1])/360:.3f} revolution)')
print(f'rotation sense        : '
      f'{"clockwise" if ang[-1] < ang[0] else "counter-clockwise"}')
print(f'period 2*pi/eps       : {2*np.pi/eps:.2f}')
print(f'decay time constant   : {1/abs(a):.2f}')

plt.show()