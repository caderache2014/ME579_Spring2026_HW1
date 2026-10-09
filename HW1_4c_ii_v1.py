"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(c), step (ii): eigenvectors of

    A = [[ a,   eps],
         [-eps, a  ]],     a = -0.1,  eps = 0.05

The eigenvalues are a complex conjugate pair, so the eigenvectors are
complex. They therefore have no direction in the real x1-x2 plane, and
the "are they close to parallel" question does not apply here.
"""

import numpy as np

a, eps = -0.1, 0.05
A = np.array([[a,    eps],
              [-eps, a  ]])

lam, vec = np.linalg.eig(A)
vec = vec / np.linalg.norm(vec, axis=0)

print(f'A =\n{A}\n')
print(f'eigenvalues            : {lam}')
print(f'eigenvectors (columns) :\n{vec}\n')
print(f'purely real            : {np.allclose(vec.imag, 0.0)}')
print('-> eigenvectors are COMPLEX, so they are not plotted in step (iv)')
print('   and the near-parallel question does not apply.\n')

# Verify the conjugate relationship between the two eigenvectors.
print(f'v2 vs conj(v1)         : {np.allclose(vec[:, 1], np.conj(vec[:, 0]))}')

# Verify A v = lambda v for the first pair.
resid = np.linalg.norm(A @ vec[:, 0] - lam[0] * vec[:, 0])
print(f'residual |A v - lam v| : {resid:.3e}\n')

# The real and imaginary parts of a complex eigenvector span the real
# invariant plane. Here they are orthogonal and equal in length, which is
# why the orbits are circles rather than slanted ellipses.
u, w = vec[:, 0].real, vec[:, 0].imag
print(f'Re(v1)                 : {u}')
print(f'Im(v1)                 : {w}')
print(f'|Re(v1)|, |Im(v1)|     : {np.linalg.norm(u):.4f}, {np.linalg.norm(w):.4f}')
print(f'dot(Re, Im)            : {np.dot(u, w):.3e}  (0 => orthogonal)')