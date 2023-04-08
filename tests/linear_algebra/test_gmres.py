import unittest, numpy as np
from aerodesk import Gmres

np.random.seed(1234567)

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
        """solve a small 10x10 system with the GMRES algorithm"""
        A = np.random.rand(100, 100)
        b = np.random.rand(100, 1)

        resid = Gmres.solve(A, b).residual
        print(f"GMRES - 100x100 residual = {resid}")
        self.assertTrue(resid < 1e-10)
        return

if __name__ == "__main__":
    unittest.main()
