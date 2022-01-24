import numpy as np


def ldas(K, F, U, FF, UU, tol=1e-6):
    for i, b in enumerate(F.T):
        if np.linalg.norm(b) < tol:
            continue
        d = []
        r = b.copy()
        for bb in FF:
            a = np.dot(b, bb) / np.dot(bb, bb)
            r -= a * bb
            d.append(a)
        if np.linalg.norm(r) < tol:
            U[:, i] = np.dot(d, np.asarray(UU))
            continue
        print(f"Solve for load: {i}...")
        FF.append(r)
        UU.append(np.linalg.solve(K, r))
        U[:, i] = UU[-1] + np.dot(d, np.asarray(UU[:-1]))
    return U
