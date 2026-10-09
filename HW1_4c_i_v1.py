"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(c), step (i): eigenvalues of

    A = [[ a, eps],
         [-eps, a ]],     a = -0.1,  eps = 0.05
"""

import numpy as np

a, eps = -0.1, 0.05
A = np.array([[a,    eps],
              [-eps, a  ]])

lam = np.linalg.eigvals(A)

tol = 1e-12
n_distinct = len(np.unique(np.round(lam / tol) * tol))

print(f'A =\n{A}\n')
print(f'eigenvalues          : {lam}')
print(f'trace                : {np.trace(A):+.4f}')
print(f'determinant          : {np.linalg.det(A):+.6f}')
print(f'discriminant tr^2-4d : {np.trace(A)**2 - 4*np.linalg.det(A):+.6f}')
print(f'distinct eigenvalues : {n_distinct}')
print()
print(f'real part            : {lam[0].real:+.4f}   (decay rate)')
print(f'imag part            : {abs(lam[0].imag):+.4f}   (rad per unit time)')
print(f'oscillation period   : {2*np.pi/abs(lam[0].imag):.4f}')
print(f'decay time constant  : {1/abs(lam[0].real):.4f}')