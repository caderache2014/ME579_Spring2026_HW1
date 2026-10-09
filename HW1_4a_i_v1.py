"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(a), step (i): eigenvalues of

    A = [[a, 0],
         [0, a]],     a = -0.1
"""

import numpy as np

a = -0.1
A = np.array([[a,   0.0],
              [0.0, a  ]])

lam = np.linalg.eigvals(A)

# numpy returns one value per dimension, repeats included. Round before
# de-duplicating so that a genuinely repeated eigenvalue is not reported
# as two distinct ones because of floating-point noise.
tol = 1e-12
n_distinct = len(np.unique(np.round(lam / tol) * tol))

print(f'A =\n{A}\n')
print(f'eigenvalues          : {lam}')
print(f'trace                : {np.trace(A):+.4f}')
print(f'determinant          : {np.linalg.det(A):+.4f}')
print(f'discriminant tr^2-4d : {np.trace(A)**2 - 4*np.linalg.det(A):+.4e}')
print(f'distinct eigenvalues : {n_distinct}')
print()
print('The discriminant is zero, so the root is repeated:')
print(f'  lambda^2 - ({np.trace(A):+.1f})lambda + ({np.linalg.det(A):.2f}) '
      f'= (lambda - ({a}))^2')
print()
print(f'decay rate           : {lam[0].real:+.4f}')
print(f'time constant        : {1/abs(lam[0].real):.1f}')