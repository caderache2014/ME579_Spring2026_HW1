"""
ME579 - Autumn 2026, Homework 1B
Exercise 1-4(e), step (i): eigenvalues of

    A = [[a,  0],
         [0, -a]],     a = -0.1
"""

import numpy as np

a = -0.1
A = np.array([[a,   0.0],
              [0.0, -a ]])

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
print('A negative determinant is by itself sufficient to identify a')
print('saddle: real eigenvalues of opposite sign.')
print()
for l in lam:
    kind = 'decays' if l.real < 0 else 'grows'
    print(f'  lambda = {l.real:+.4f}  -> {kind} like exp({l.real:+.4f} t), '
          f'time constant {1/abs(l.real):.1f}')