"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4: analysis of dx/dt = A x for five matrices.

Step (i): eigenvalues and how many are distinct.

This module holds all five matrices so later parts can reuse them.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import expm

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                     'axes.labelsize': 12, 'legend.fontsize': 11})

a, eps = -0.1, 0.05

MATRICES = {
    '(a)': np.array([[a,    0.0],
                     [0.0,  a  ]]),
    '(b)': np.array([[a,    eps],
                     [0.0,  a  ]]),
    '(c)': np.array([[a,    eps],
                     [-eps, a  ]]),
    '(d)': np.array([[0.0,  a  ],
                     [-a,   0.0]]),
    '(e)': np.array([[a,    0.0],
                     [0.0, -a  ]]),
}


def eigenvalues(A, tol=1e-12):
    """
    Eigenvalues of A, plus a count of how many are DISTINCT.

    numpy always returns n eigenvalues, repeats included, so the distinct
    count is obtained by rounding and de-duplicating. The tolerance guards
    against two genuinely equal eigenvalues being reported as different in
    the last bits of floating point.
    """
    lam = np.linalg.eigvals(A)
    rounded = np.round(lam / tol) * tol
    n_distinct = len(np.unique(rounded))
    return lam, n_distinct


def eigenvectors(A):
    """
    Eigenvalues and unit eigenvectors of A, with diagnostics.

    Returns the eigen-decomposition, whether the eigenvectors are purely
    real, the angle between them, and the rank of (A - lambda*I) for a
    repeated eigenvalue, which is what distinguishes a full 2-D
    eigenspace from a defective 1-D one.
    """
    lam, vec = np.linalg.eig(A)
    vec = vec / np.linalg.norm(vec, axis=0)          # unit length
    purely_real = np.allclose(vec.imag, 0.0)

    angle = None
    if purely_real:
        v1, v2 = vec[:, 0].real, vec[:, 1].real
        # Absolute value: v and -v span the same direction, so the angle
        # between DIRECTIONS is taken in [0, 90] degrees.
        c = np.clip(abs(np.dot(v1, v2)), 0.0, 1.0)
        angle = np.degrees(np.arccos(c))

    # Geometric multiplicity of the first eigenvalue.
    n_eigvec = A.shape[0] - np.linalg.matrix_rank(A - lam[0] * np.eye(A.shape[0]))
    return lam, vec, purely_real, angle, n_eigvec


def simulate(A, x0=(0.2, 0.5), t_end=100.0, n=4000):
    """Integrate dx/dt = A x with solve_ivp."""
    rhs = lambda t, x: A @ x
    sol = solve_ivp(rhs, (0.0, t_end), list(x0),
                    t_eval=np.linspace(0.0, t_end, n),
                    rtol=1e-10, atol=1e-12)
    return sol.t, sol.y


def plot_time_series(A, label, x0=(0.2, 0.5), t_end=100.0, fname=None):
    """Plot x1(t) and x2(t), with the matrix-exponential solution overlaid
    as a dashed line to verify the numerical integration."""
    t, y = simulate(A, x0, t_end)

    # Exact solution x(t) = expm(A t) x0, evaluated on a coarse grid.
    t_chk = np.linspace(0.0, t_end, 25)
    exact = np.array([expm(A * ti) @ np.array(x0) for ti in t_chk]).T

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, y[0], color='tab:blue', lw=1.8, label='$x_1(t)$')
    ax.plot(t, y[1], color='tab:orange', lw=1.8, label='$x_2(t)$')
    ax.plot(t_chk, exact[0], 'k.', ms=5, label=r'exact, $e^{At}x_0$')
    ax.plot(t_chk, exact[1], 'k.', ms=5)

    ax.axhline(0, color='0.8', lw=0.8, zorder=0)
    ax.set_xlim(0, t_end)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('state')
    ax.set_title(rf'Exercise 1-4{label}:  $x_0=[0.2,\,0.5]^T$')
    ax.legend(loc='upper right', framealpha=0.95)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200, bbox_inches='tight')
    return t, y


def plot_phase_portrait(A, label, x0=(0.2, 0.5), lim=0.6, t_end=100.0,
                        fname=None, show_eigenspace=False):
    """
    Phase portrait of x1 vs x2, with purely real eigenvectors overlaid.

    show_eigenspace=True additionally draws faint radial lines, used for
    A = a*I where EVERY direction is an eigendirection and any single pair
    of eigenvectors would misleadingly suggest two special axes.
    """
    fig, ax = plt.subplots(figsize=(7, 6.5))

    g = np.linspace(-lim, lim, 25)
    X1, X2 = np.meshgrid(g, g)
    dX1 = A[0, 0] * X1 + A[0, 1] * X2
    dX2 = A[1, 0] * X1 + A[1, 1] * X2
    ax.streamplot(X1, X2, dX1, dX2, density=1.2, color='0.65',
                  linewidth=0.7, arrowsize=0.9)

    if show_eigenspace:
        for ang in np.arange(0, 180, 15):
            d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            ax.plot([-lim*d[0], lim*d[0]], [-lim*d[1], lim*d[1]],
                    color='tab:green', lw=0.8, alpha=0.35, zorder=2)

    lam, vec, purely_real, _, _ = eigenvectors(A)
    if purely_real:
        colors = ['tab:green', 'tab:purple']
        for k in range(vec.shape[1]):
            v = vec[:, k].real
            ax.plot([-lim*v[0], lim*v[0]], [-lim*v[1], lim*v[1]],
                    color=colors[k], lw=2.2, zorder=3,
                    label=rf'$v_{k+1}$, $\lambda={lam[k].real:+.2f}$')

    t, y = simulate(A, x0, t_end)
    ax.plot(y[0], y[1], color='tab:blue', lw=2.4, zorder=5,
            label='trajectory')
    ax.plot(*x0, 'o', color='tab:blue', ms=8, mec='k', mew=0.8, zorder=6,
            label=r'$x_0=[0.2,\,0.5]^T$')
    ax.plot(0, 0, 'o', color='k', ms=9, zorder=6)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Exercise 1-4{label}: phase portrait')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200, bbox_inches='tight')
    return fig, ax


if __name__ == '__main__':
    for label in ['(a)']:
        A = MATRICES[label]
        lam, n = eigenvalues(A)
        print(f'Matrix {label}:')
        print(f'  A =\n{A}')
        print(f'  eigenvalues         : {lam}')
        print(f'  trace               : {np.trace(A):+.4f}')
        print(f'  determinant         : {np.linalg.det(A):+.4f}')
        print(f'  distinct eigenvalues: {n}')

        lam, vec, real, angle, geo = eigenvectors(A)
        print(f'  eigenvectors (cols) :\n{vec}')
        print(f'  purely real         : {real}')
        print(f'  angle between them  : '
              f'{angle:.2f} deg' if angle is not None else '  (complex)')
        print(f'  dim of eigenspace   : {geo}')

        tag = label.strip('()')
        t, y = plot_time_series(A, label,
                                fname=f'ex1_4{tag}_timeseries.png')
        print(f'  x(0)                : [{y[0][0]:.6f}, {y[1][0]:.6f}]')
        print(f'  x(100)              : [{y[0][-1]:.3e}, {y[1][-1]:.3e}]')
        print(f'  ratio x2/x1 at t=0  : {y[1][0]/y[0][0]:.6f}')
        print(f'  ratio x2/x1 at t=100: {y[1][-1]/y[0][-1]:.6f}')

        # A = a*I has a full 2-D eigenspace, so show the radial family.
        plot_phase_portrait(A, label, fname=f'HW1_4{tag}_phase.png',
                            show_eigenspace=(tag == 'a'))

    plt.show()