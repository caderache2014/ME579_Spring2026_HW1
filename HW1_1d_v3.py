"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-1(d): Phase portraits of the damped pendulum.

Nonlinear system (with l = m = g = b = 1):
    theta_dot = omega
    omega_dot = -sin(theta) - omega

Linearization about the origin (theta* = 0, pendulum hanging down):
    d/dt [d_theta, d_omega]^T = A @ [d_theta, d_omega]^T
    A = J(0) = [[0, 1], [-1, -1]]

The plot window spans theta in [-2*pi, 2*pi], which contains FIVE nonlinear
fixed points: (-2pi, 0), (-pi, 0), (0, 0), (pi, 0), (2pi, 0).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 10})
# ----------------------------------------------------------------------
# 1. System definitions
# ----------------------------------------------------------------------

# Jacobian evaluated at the downward equilibrium, theta* = 0.
# J(theta) = [[0, 1], [-cos(theta), -1]]  ->  J(0) = [[0, 1], [-1, -1]]
A_LIN = np.array([[0.0, 1.0],
                  [-1.0, -1.0]])


def jacobian(theta):
    """Jacobian of the nonlinear pendulum evaluated at angle theta."""
    return np.array([[0.0, 1.0],
                     [-np.cos(theta), -1.0]])


def nonlinear_rhs(t, state):
    """Right-hand side of the full nonlinear system. Used by solve_ivp."""
    theta, omega = state
    return [omega, -np.sin(theta) - omega]


def linear_rhs(t, state):
    """Right-hand side of the linearized system (valid near theta = 0)."""
    return A_LIN @ np.asarray(state)


def nonlinear_field(theta, omega):
    """Vectorized vector field for streamplot (nonlinear)."""
    return omega, -np.sin(theta) - omega


def linear_field(theta, omega):
    """Vectorized vector field for streamplot (linearized)."""
    dtheta = A_LIN[0, 0] * theta + A_LIN[0, 1] * omega
    domega = A_LIN[1, 0] * theta + A_LIN[1, 1] * omega
    return dtheta, domega


# ----------------------------------------------------------------------
# 2. Plot window and fixed points
# ----------------------------------------------------------------------

THETA_MIN, THETA_MAX = -2 * np.pi, 2 * np.pi
OMEGA_MIN, OMEGA_MAX = -4.0, 4.0

# Fixed points at (n*pi, 0). Even n -> hanging down (stable spiral),
# odd n -> inverted (saddle).
N_RANGE = range(-2, 3)
STABLE_FPS = [(n * np.pi, 0.0) for n in N_RANGE if n % 2 == 0]
SADDLE_FPS = [(n * np.pi, 0.0) for n in N_RANGE if n % 2 != 0]


# ----------------------------------------------------------------------
# 3. Helper routines
# ----------------------------------------------------------------------

def integrate(rhs, x0, t_end, n_points=4000):
    """Integrate `rhs` from state x0. Negative t_end integrates backward."""
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(rhs, (0.0, t_end), x0, t_eval=t_eval,
                    rtol=1e-9, atol=1e-11, dense_output=True)
    return sol.y[0], sol.y[1]


def saddle_manifolds(theta_star, eps=1e-4, t_unstable=30.0, t_stable=10.0):
    """
    Numerically trace the stable and unstable manifolds (the separatrices)
    of the saddle at (theta_star, 0).

    The unstable manifold is found by integrating FORWARD from a point
    displaced slightly along the unstable eigenvector. The stable manifold
    is found by integrating BACKWARD along the stable eigenvector, which
    turns the attracting direction into a repelling one.

    The two horizons differ because the eigenvalues differ in magnitude.
    Starting a distance eps away, the departure grows like exp(|lambda| t),
    so the slow unstable direction (lambda = +0.618) needs roughly 2.6x
    longer to travel the same distance as the fast stable one (-1.618).
    """
    J = jacobian(theta_star)
    evals, evecs = np.linalg.eig(J)

    i_unstable = int(np.argmax(evals.real))
    i_stable = int(np.argmin(evals.real))
    v_u = np.real(evecs[:, i_unstable])
    v_s = np.real(evecs[:, i_stable])

    x_star = np.array([theta_star, 0.0])
    unstable, stable = [], []
    for sign in (+1, -1):
        unstable.append(integrate(nonlinear_rhs, x_star + sign * eps * v_u, +t_unstable))
        stable.append(integrate(nonlinear_rhs, x_star + sign * eps * v_s, -t_stable))
    return unstable, stable, v_u, v_s, evals.real


def draw_fixed_points(ax, stable_pts, saddle_pts):
    """Filled circles for stable spirals, open squares for saddles."""
    if stable_pts:
        ax.plot(*zip(*stable_pts), 'o', color='k', ms=8, zorder=6,
                label='stable spiral')
    if saddle_pts:
        ax.plot(*zip(*saddle_pts), 's', mfc='white', mec='k', mew=1.8,
                ms=8, zorder=6, label='saddle')


def style_axes(ax, title):
    ax.set_xlim(THETA_MIN, THETA_MAX)
    ax.set_ylim(OMEGA_MIN, OMEGA_MAX)
    ax.set_xlabel(r'$\theta$  [rad]')
    ax.set_ylabel(r'$\omega = \dot{\theta}$  [rad/s]')
    ax.set_title(title)
    ax.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
    ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$'])
    ax.axhline(0, color='0.8', lw=0.8, zorder=0)
    ax.axvline(0, color='0.8', lw=0.8, zorder=0)


# ----------------------------------------------------------------------
# 4. Build the figure
# ----------------------------------------------------------------------

fig, (ax_lin, ax_non) = plt.subplots(2, 1, figsize=(8, 8.5))

# Background grid for streamplot. The sample code in the assignment uses 20
# points per axis; a denser grid gives smoother streamlines.
theta_g, omega_g = np.meshgrid(np.linspace(THETA_MIN, THETA_MAX, 60),
                               np.linspace(OMEGA_MIN, OMEGA_MAX, 60))

# ---- Left panel: linearized system ----------------------------------
dth_l, dom_l = linear_field(theta_g, omega_g)
ax_lin.streamplot(theta_g, omega_g, dth_l, dom_l,
                  density=1.3, color='0.55', linewidth=0.8, arrowsize=0.9)
draw_fixed_points(ax_lin, [(0.0, 0.0)], [])
for x0 in [(2.0, 0.0), (-2.0, 0.0), (0.0, 3.0), (5.0, 0.0)]:
    th, om = integrate(linear_rhs, list(x0), 30.0)
    ax_lin.plot(th, om, lw=1.8, color='tab:blue')
    ax_lin.plot(x0[0], x0[1], 'o', color='tab:blue', ms=4)
style_axes(ax_lin, 'Linearized system:  '
                   r'$\dot{\theta}=\omega,\ \dot{\omega}=-\theta-\omega$')
ax_lin.legend(loc='upper right', framealpha=0.95)
ax_lin.set_xlabel('')
ax_lin.tick_params(labelbottom=False)

# ---- Right panel: nonlinear system ----------------------------------
dth_n, dom_n = nonlinear_field(theta_g, omega_g)
ax_non.streamplot(theta_g, omega_g, dth_n, dom_n,
                  density=1.3, color='0.55', linewidth=0.8, arrowsize=0.9)

# Separatrices emanating from every saddle in view.
printed = False
for theta_star, _ in SADDLE_FPS:
    unstable, stable, v_u, v_s, lam = saddle_manifolds(theta_star)
    for th, om in unstable:
        ax_non.plot(th, om, color='tab:red', lw=1.6, zorder=4)
    for th, om in stable:
        ax_non.plot(th, om, color='tab:green', lw=1.6, zorder=4)
    # Draw the eigenvector directions at every saddle in the window.
    for v, c in ((v_u, 'tab:red'), (v_s, 'tab:green')):
        v = v / np.linalg.norm(v)
        ax_non.annotate('', xy=(theta_star + 1.1 * v[0], 1.1 * v[1]),
                        xytext=(theta_star - 1.1 * v[0], -1.1 * v[1]),
                        arrowprops=dict(arrowstyle='<->', color=c, lw=2.2),
                        zorder=7)
    # The eigen-decomposition is identical at every saddle, so report once.
    if not printed:
        print(f'Saddle at theta = {theta_star:+.4f}:')
        print(f'   eigenvalues      : {lam[0]:+.4f}, {lam[1]:+.4f}')
        print(f'   unstable eigvec  : [{v_u[0]:+.4f}, {v_u[1]:+.4f}]')
        print(f'   stable   eigvec  : [{v_s[0]:+.4f}, {v_s[1]:+.4f}]')
        printed = True

# A few representative trajectories, including one energetic enough to
# swing past the top before settling into a different well.
for x0, col in [((0.5, 0.0), 'tab:blue'),
                ((2.5, 0.0), 'tab:blue'),
                ((-np.pi + 0.3, 0.0), 'tab:blue'),
                ((-2.0, 3.6), 'tab:purple'),
                ((-4.0, 3.2), 'tab:purple')]:
    th, om = integrate(nonlinear_rhs, list(x0), 40.0)
    ax_non.plot(th, om, lw=1.8, color=col, zorder=5)
    ax_non.plot(x0[0], x0[1], 'o', color=col, ms=4, zorder=5)

draw_fixed_points(ax_non, STABLE_FPS, SADDLE_FPS)
style_axes(ax_non, 'Nonlinear system:  '
                   r'$\dot{\theta}=\omega,\ \dot{\omega}=-\sin\theta-\omega$')

handles, labels = ax_non.get_legend_handles_labels()
handles += [plt.Line2D([], [], color='tab:red', lw=1.6),
            plt.Line2D([], [], color='tab:green', lw=1.6)]
labels += ['unstable manifold', 'stable manifold (separatrix)']
ax_non.legend(handles, labels, loc='upper right', framealpha=0.95, fontsize=10)

fig.suptitle('ME579 HW1B, Ex. 1-1(d): damped pendulum phase portraits '
             r'($l=m=g=b=1$)', fontsize=13)
fig.tight_layout()
fig.savefig('HW1_1d_phase_portraits_v3.png', dpi=200, bbox_inches='tight')

# Report the eigenvalues at the origin for cross-checking part (c).
lam0 = np.linalg.eigvals(A_LIN)
print('\nOrigin (hanging down), eigenvalues of A = J(0):')
for l in lam0:
    print(f'   {l.real:+.4f} {l.imag:+.4f}j')

plt.show()