"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(b), step (ii): eigenvectors of

    A = [[a, eps],
         [0, a  ]],     a = -0.1,  eps = 0.05

The eigenvalue is repeated, so the question is how many INDEPENDENT
eigenvectors exist. numpy will still return a 2x2 array of "eigenvectors";
the check below shows whether its two columns are actually independent.
"""

import numpy as np

a, eps = -0.1, 0.05
A = np.array([[a,   eps],
              [0.0, a  ]])

lam, vec = np.linalg.eig(A)
vec = vec / np.linalg.norm(vec, axis=0)          # unit length

v1, v2 = vec[:, 0].real, vec[:, 1].real

# Angle between the two DIRECTIONS returned. abs() because v and -v span
# the same line, so the angle is taken in [0, 90] degrees.
angle = np.degrees(np.arccos(np.clip(abs(np.dot(v1, v2)), 0.0, 1.0)))

print(f'A =\n{A}\n')
print(f'eigenvalues            : {lam}')
print(f'eigenvectors (columns) :\n{vec.real}\n')
print(f'purely real            : {np.allclose(vec.imag, 0.0)}')
print(f'angle between them     : {angle:.4e} deg')
print(f'rank of eigenvector mx : {np.linalg.matrix_rank(vec.real)}')
print(f'dim of eigenspace      : {2 - np.linalg.matrix_rank(A - lam[0]*np.eye(2))}')

# Generalized eigenvector w, solving (A - lambda I) w = v1.
w = np.linalg.lstsq(A - lam[0] * np.eye(2), v1, rcond=None)[0]
print(f'\ngeneralized eigenvector: {w}')
print(f'check (A-lam I) w = v1 : {(A - lam[0]*np.eye(2)) @ w}  vs  {v1}')