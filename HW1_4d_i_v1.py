"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(d), step (i): eigenvalues of

    A = [[ 0, a],
         [-a, 0]],     a = -0.1
"""

import numpy as np

a = -0.1
A = np.array([[0.0, a  ],
              [-a,  0.0]])

lam = np.linalg.eigvals(A)

tol = 1e-12
n_distinct = len(np.unique(np.round(lam / tol) * tol))

print(f'A =\n{A}\n')
print(f'eigenvalues          : {lam}')
print(f'trace                : {np.trace(A):+.4f}')
print(f'determinant          : {np.linalg.det(A):+.4f}')
print(f'discriminant tr^2-4d : {np.trace(A)**2 - 4*np.linalg.det(A):+.4f}')
print(f'distinct eigenvalues : {n_distinct}')
print()
print(f'real part            : {lam[0].real:+.4f}   (no decay or growth)')
print(f'imag part            : {abs(lam[0].imag):+.4f}   (rad per unit time)')
print(f'oscillation period   : {2*np.pi/abs(lam[0].imag):.4f}')
print()
# A^T = -A, so the matrix is skew-symmetric: a pure rotation generator.
print(f'skew-symmetric (A^T = -A)? {np.allclose(A.T, -A)}')