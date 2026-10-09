"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(d), step (ii): eigenvectors of

    A = [[ 0, a],
         [-a, 0]],     a = -0.1

The eigenvalues are a purely imaginary conjugate pair, so the
eigenvectors are complex. They define no direction in the real
x1-x2 plane, and the "are they close to parallel" question does
not apply.
"""

import numpy as np

a = -0.1
A = np.array([[0.0, a  ],
              [-a,  0.0]])

lam, vec = np.linalg.eig(A)
vec = vec / np.linalg.norm(vec, axis=0)

print(f'A =\n{A}\n')
print(f'eigenvalues            : {lam}')
print(f'eigenvectors (columns) :\n{vec}\n')
print(f'purely real            : {np.allclose(vec.imag, 0.0)}')
print('-> complex, so not plotted in step (iv) and the')
print('   near-parallel question does not apply.\n')

print(f'v2 vs conj(v1)         : {np.allclose(vec[:, 1], np.conj(vec[:, 0]))}')
resid = np.linalg.norm(A @ vec[:, 0] - lam[0] * vec[:, 0])
print(f'residual |A v - lam v| : {resid:.3e}\n')

# Real and imaginary parts span the real invariant plane. Equal length and
# orthogonal => circular orbits rather than slanted ellipses.
u, w = vec[:, 0].real, vec[:, 0].imag
print(f'Re(v1)                 : {u}')
print(f'Im(v1)                 : {w}')
print(f'|Re(v1)|, |Im(v1)|     : {np.linalg.norm(u):.4f}, {np.linalg.norm(w):.4f}')
print(f'dot(Re, Im)            : {np.dot(u, w):.3e}  (0 => orthogonal)\n')

# Conserved quantity: for skew-symmetric A, d/dt |x|^2 = 2 x.(A x) = 0.
x = np.array([0.2, 0.5])
print(f'x . (A x) for x = {x}: {np.dot(x, A @ x):.3e}')
print('-> |x| is exactly conserved, so orbits are circles.')