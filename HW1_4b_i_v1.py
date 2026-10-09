"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(b), step (i): eigenvalues of

    A = [[a, eps],
         [0, a  ]],     a = -0.1,  eps = 0.05
"""

import numpy as np

a, eps = -0.1, 0.05
A = np.array([[a,   eps],
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

# Geometric multiplicity: dim of the null space of (A - lambda I).
rank = np.linalg.matrix_rank(A - lam[0] * np.eye(2))
print(f'rank(A - lambda I)   : {rank}')
print(f'dim of eigenspace    : {2 - rank}')