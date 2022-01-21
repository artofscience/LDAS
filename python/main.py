import numpy as np

from alg import AXB

A = np.array([[2, 0],
              [0, 2]], dtype=float)

B = np.array([[0, 1, 2, 0, 0, 5],
              [0, 0, 2, 0, 1, 2]], dtype=float)

X = np.zeros_like(B)
Bbasis = []
Xbasis = []

AXB(A, B, X, Bbasis, Xbasis)

print(Bbasis)
print(Xbasis)
print(B)
print(X)

