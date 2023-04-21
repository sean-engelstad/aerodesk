import unittest, numpy as np
from aerodesk import Gmres

# np.random.seed(134567)


class GmresSolverTest(unittest.TestCase):
    def test_10x10(self):
        """solve a 10x10 real system with the GMRES algorithm"""
        N = 10
        A = np.random.rand(N, N)
        b = np.random.rand(N, 1)

        resid = Gmres.solve(A, b).residual
        print(f"GMRES - {N}x{N} real residual = {resid}")
        self.assertTrue(resid < 1e-10)
        return

    def test_100x100(self):
        """solve a 100x100 real system with the GMRES algorithm"""
        N = 100
        A = np.random.rand(N, N)
        b = np.random.rand(N, 1)

        resid = Gmres.solve(A, b).residual
        print(f"GMRES - {N}x{N} real residual = {resid}")
        self.assertTrue(resid < 1e-8)
        return

    def test_10x10_complex(self):
        """solve a 10x10 complex system with the GMRES algorithm"""
        N = 10
        A = np.random.rand(N, N) + np.random.rand(N, N) * 1e-5 * 1j
        b = np.random.rand(N, 1) + np.random.rand(N, 1) * 1e-5 * 1j

        resid = Gmres.solve(A, b, complex=True).residual
        print(f"GMRES - {N}x{N} complex residual = {resid}")
        self.assertTrue(resid < 1e-8)
        return

    def test_100x100_complex(self):
        """solve a 100x100 complex system with the GMRES algorithm"""
        N = 100
        A = np.random.rand(N, N) + np.random.rand(N, N) * 1e-5 * 1j
        b = np.random.rand(N, 1) + np.random.rand(N, 1) * 1e-5 * 1j

        resid = Gmres.solve(A, b, tol=1e-14, complex=True).residual
        print(f"GMRES - {N}x{N} complex residual = {resid}")
        self.assertTrue(resid < 1e-6)
        return

    # @unittest.skip("not working failing at 1e-6 now")
    def test_500x500_real(self):
        """test large 500x500 real system with GMRES"""
        N = 500
        A = np.random.rand(N, N)
        b = np.random.rand(N, 1)

        # resid = Gmres.solve(A, b).residual
        solver = Gmres.solve(A, b, tol=1e-20)
        resid = solver.residual
        # resid_vec = b - A @ solver.x
        # print(f"for N = {N}, r = {resid_vec}")
        print(f"GMRES - {N}x{N} real residual = {resid}")
        self.assertTrue(resid < 1e-8)
        return


if __name__ == "__main__":
    unittest.main()
