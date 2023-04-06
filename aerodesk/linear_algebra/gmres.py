
__all__ = ["GmresSolver"]

import numpy as np

class GmresSolver:
    """https://stackoverflow.com/questions/37962271/whats-wrong-with-my-gmres-implementation"""
    def __init__(self, A, b, maxiter, x0=None, tol=1e-10):
        self.A = A
        self.b = b
        self.maxiter = maxiter
        self.x0 = x0 if x0 is not None else b
        self.tol = tol
        
        # check that A is a square matrix
        Ashape = self.A.shape
        assert(Ashape[0] == Ashape[1])
        self.N = Ashape[0]

        bshape = self.b.shape
        assert(len(bshape) == 2 and bshape[1] == 1)

        # initialize helper matricesm H,Q from Arnoldi
        self.Q = np.zeros((self.N, maxiter))
        self.H = np.zeros((maxiter+1, maxiter))

    def solve(self):
        """solve the matrix with the desired GMRES algorithm"""
        x = self.x0
        r = self.b - self.A @ x
        q0 = r / np.linalg.norm(r)
        self.Q[:,0] = q0[:,0]

        for k in range(self.maxiter):
            qnext = self.A @ self.Q[:,k]

            for j in range(k):
                self.H[j,k] = qnext.T @ self.Q[:,j]
                qnext -= self.H[j,k] * self.Q[:,j]
            self.H[k+1,k] = np.linalg.norm(qnext)
            if self.H[k+1,k] != 0 and k < self.maxiter-1:
                qnext /= self.H[k+1,k]
                self.Q[:,k] = qnext[:,0]
            else:
                break
            if 