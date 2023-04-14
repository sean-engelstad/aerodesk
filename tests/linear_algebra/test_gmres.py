import unittest, numpy as np
from aerodesk import Gmres

# np.random.seed(134567)


class GmresSolverTest(unittest.TestCase):
    def test_10x10(self):
        """solve a small 10x10 system with the GMRES algorithm"""
        A = np.random.rand(10, 10)
        b = np.random.rand(10, 1)

        resid = Gmres.solve(A, b).residual
        print(f"GMRES - 10x10 residual = {resid}")
        self.assertTrue(resid < 1e-10)
        return

    def test_100x100(self):
        """solve a 100x100 real system with the GMRES algorithm"""
        A = np.random.rand(100, 100)
        b = np.random.rand(100, 1)

        resid = Gmres.solve(A, b).residual
        print(f"GMRES - 100x100 residual = {resid}")
        self.assertTrue(resid < 1e-8)
        return

    # @unittest.skip("complex first")
    def test_100x100_complex(self):
        """solve a 100x100 complex system with the GMRES algorithm"""
        A = np.random.rand(100, 100) + np.random.rand(100, 100) * 1e-5 * 1j
        b = np.random.rand(100, 1) + np.random.rand(100, 1) * 1e-5 * 1j
        print(f"A = {A}")

        resid = Gmres.solve(A, b, complex=True).residual
        print(f"GMRES - 100x100 complex residual = {resid}")
        self.assertTrue(resid < 1e-8)
        return


if __name__ == "__main__":
    unittest.main()
