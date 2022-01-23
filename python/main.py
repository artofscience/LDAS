import numpy as np

from ldas import ldas

K = np.array([[2, 0],
              [0, 2]], dtype=float)

F = np.array([[0, 1, 2, 0, 0, 5],
              [0, 0, 2, 0, 1, 2]], dtype=float)

U = np.zeros_like(F)
F_basis = []
U_basis = []

ldas(K, F, U, F_basis, U_basis)

print(F)
print(U)
print(F_basis)
print(U_basis)
