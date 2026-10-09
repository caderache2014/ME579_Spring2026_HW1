"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(e), step (ii): eigenvectors of

    A = [[a,  0],
         [0, -a]],     a = -0.1

The eigenvalues are real and distinct, so the eigenvectors are purely
real and ARE plotted in step (iv). The matrix is already diagonal, so
they are the coordinate axes.
"""

import numpy as np

a = -0.1
A = np.array([[a,   0.0],
              [0.0, -a ]])

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

for k in range(2):
    v = vec[:, k].real
    role = 'STABLE  (decaying)' if lam[k].real < 0 else 'UNSTABLE (growing)'
    print(f'  lambda = {lam[k].real:+.2f}  v = [{v[0]:+.0f}, {v[1]:+.0f}]  '
          f'-> {role}')
print()

# Verify the eigendirections are invariant: starting on one, you stay on it.
for k in range(2):
    v = vec[:, k].real
    print(f'A v{k+1} = {A @ v}  =  {lam[k].real:+.2f} * v{k+1}  '
          f'-> {np.allclose(A @ v, lam[k].real * v)}')