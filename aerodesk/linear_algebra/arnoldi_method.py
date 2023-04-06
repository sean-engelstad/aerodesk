__all__ = ["ArnoldisMethod"]

import numpy as np


class ArnoldisMethod:
    def __init__(self, A, nsteps, q0):
        """
        takes in a matrix A and builds a basis for the Krylov subspace with the Graham-Schmidt process

        Parameters
        ----------
        A : np.ndarray
            the square matrix to factorize with Arnoldi
        nsteps : int
            the number of elements in the basis or steps in the algorithm
        q0 : np.ndarray
            the column unit vector to which starts the Krylov subspace
        """
        self.A = A
        self.nsteps = nsteps

        # check square matrix
        Ashape = self.A.shape
        assert Ashape[0] == Ashape[1]
        self.N = Ashape[0]

        # initialize Q and H matrices
        self.Q = np.zeros((self.N, nsteps))
        self.H = np.zeros((nsteps + 1, nsteps))

        self.Q[:, 0] = q0[:, 0]

        # iteration counter
        self.k = 0

    def iterate(self):
        """perform the next Arnoldi iteration"""
        # previous basis vector
        k = self.k
        q = self.Q[:, k]
        qnext = self.A @ q
        for i in range(k + 1):  # 0,1,2,...,k
            self.H[i, k] = qnext.T @ self.Q[:, i]
        for i in range(k + 1):
            qnext[:] -= self.H[i, k] * self.Q[:, i]
        self.H[k + 1, k] = np.linalg.norm(qnext)
        self.Q[:, k + 1] = qnext / self.H[k + 1, k]
        return

    def check_identity(self):
        """check the Arnoldi's recursion identity A * Q_k = Q_k+1 * H_k+1,k"""
        LHS = self.A @ self.Q[:, : self.nsteps]
        RHS = self.Q[:, : self.nsteps + 1] @ self.H[: self.nsteps + 1, : self.nsteps]
        diff = LHS - RHS
        arnoldi_norm = np.linalg.norm(diff)
        return arnoldi_norm
