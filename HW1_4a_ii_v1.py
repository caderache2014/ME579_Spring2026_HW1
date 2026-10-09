"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(a), step (ii): eigenvectors of

    A = [[a, 0],
         [0, a]],     a = -0.1

A = a*I, so (A - lambda*I) is the ZERO matrix and every nonzero vector
in the plane is an eigenvector. The eigenspace is two-dimensional, which
makes this the one repeated-eigenvalue case in the exercise that is NOT
defective. Compare matrix (b), which shares the eigenvalue but has only
a one-dimensional eigenspace.
"""

import numpy as np

a = -0.1
A = np.array([[a,   0.0],
              [0.0, a  ]])

lam, vec = np.linalg.eig(A)
vec = vec / np.linalg.norm(vec, axis=0)

v1, v2 = vec[:, 0].real, vec[:, 1].real
angle = np.degrees(np.arccos(np.clip(abs(np.dot(v1, v2)), 0.0, 1.0)))

print(f'A =\n{A}\n')
print(f'eigenvalues            : {lam}')
print(f'eigenvectors (columns) :\n{vec.real}\n')
print(f'purely real            : {np.allclose(vec.imag, 0.0)}')
print(f'angle between them     : {angle:.2f} deg')
print(f'rank of eigenvector mx : {np.linalg.matrix_rank(vec.real)}')
print()

n = A.shape[0]
rank = np.linalg.matrix_rank(A - lam[0] * np.eye(n))
print(f'A - lambda*I =\n{A - lam[0]*np.eye(n)}')
print(f'rank(A - lambda I)     : {rank}')
print(f'dim of eigenspace      : {n - rank}   (geometric multiplicity)')
print(f'algebraic multiplicity : 2')
print('-> geometric = algebraic, so A is diagonalizable, NOT defective.')
print()

# Every direction is invariant: A v = lambda v for an arbitrary vector.
rng = np.random.default_rng(0)
print('Check that an arbitrary direction is also an eigenvector:')
for _ in range(3):
    v = rng.normal(size=2)
    v /= np.linalg.norm(v)
    print(f'  v = [{v[0]:+.4f}, {v[1]:+.4f}]   '
          f'A v = {a} * v  ->  {np.allclose(A @ v, a * v)}')