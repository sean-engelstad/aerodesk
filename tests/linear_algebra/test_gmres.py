import unittest, numpy as np
from aerodesk import GmresSolver

np.random.seed(1234567)


class GmresSolverTest(unittest.TestCase):
    def test_small_solve(self):
        """solve a small 10x10 system with the GMRES algorithm"""
        case = 1
        if case == 1:
            A = np.random.rand(10, 10)
            b = np.random.rand(10, 1)
            x0 = np.random.rand(10, 1)

            x = GmresSolver(A, b, maxiter=15, x0=x0).solve()
        else:
            A = np.random.rand(10, 10)
            b = np.random.rand(10)
            x0 = np.random.rand(10)

            e = 0
            nmax_iter = 15

            x = GmresSolver(A, b, maxiter=15, x0=x0).solve2()

        print(f"case {case}: ======\n")
        print(f"A = {A}")
        print(f"b = {b}")
        print(f"x = {x}")
        # _GMRes(A, b, x0, e, nmax_iter)
        # r = b - np.asarray(np.dot(A, x)).reshape(-1)
        r = b - A @ x
        rnorm = np.linalg.norm(r)
        print(f"rnorm = {rnorm}")
        self.assertTrue(rnorm < 1e-9)


if __name__ == "__main__":
    unittest.main()
