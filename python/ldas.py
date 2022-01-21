import numpy as np


def ldas(A, B, X, BB, XX, tol=1e-6):
    for i, b in enumerate(B.T):
        if np.linalg.norm(b) < tol:
            continue
        d = []
        r = b.copy()
        for bb in BB:
            a = np.dot(b, bb) / np.dot(bb, bb)
            r -= a * bb
            d.append(a)
        if np.linalg.norm(r) < tol:
            X[:, i] = np.dot(d, np.asarray(XX))
            continue
        print("solve for load {:d}".format(i))
        BB.append(r)
        XX.append(np.linalg.solve(A, r))
        X[:, i] = XX[-1] + np.dot(d, np.asarray(XX[:-1]))
    return X

