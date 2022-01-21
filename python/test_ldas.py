import alg
import numpy as np
import unittest


class TestAlg(unittest.TestCase):

    def solve(self, A, B, B_, X_):
        expected = np.linalg.solve(A, B)
        X = np.zeros_like(B)
        result = alg.AXB(A, B, X, B_, X_)

        for i in range(B.shape[1]):
            msg = ('\nExpect:\t{expected[:, i]},\nResult:\t{result[:, i]}'
                   '\nFor A={A} with rhs={B[:, i]}')

            # rel/abs tol defined here for clarity, these are the same as
            # used by `np.allclose` by default
            rtol, atol = 1e-5, 1e-8
            success = np.allclose(expected[:, i], result[:, i], rtol, atol)
            self.assertTrue(success, msg)

    def test_unit_diagonal_matrix(self):
        A = np.array([[1, 0], [0, 1]], dtype=float)
        B = np.array([[1, 2, 0, 1, 2, 6, 1], [0, 0, 1, 1, 2, 5, 4]], dtype=float)
        self.solve(A, B, [], [])

    def test_scaled_diagonal_matrix(self):
        A = np.array([[2, 0], [0, 2]], dtype=float)
        B = np.array([[1, 2, 0, 1, 2, 6, 1], [0, 0, 1, 1, 2, 5, 4]], dtype=float)
        self.solve(A, B, [], [])

    def test_random_dense_matrix(self):
        A = np.random.rand(2, 2)
        A = A*A.T # ensure symmetric, pos definite (?)
        B = np.array([[1, 2, 0, 1, 2, 6, 1], [0, 0, 1, 1, 2, 5, 4]], dtype=float)
        self.solve(A, B, [], [])

    def test_random_dense_load(self):
        for n in range(1, 10):
            A = np.random.rand(n, n)
            A = A*A.T
            B = np.random.rand(n, n)
            self.solve(A, B, [], [])


if __name__ == '__main__':
    unittest.main()

