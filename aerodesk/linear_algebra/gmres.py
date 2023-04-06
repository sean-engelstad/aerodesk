__all__ = ["GmresSolver"]

import numpy as np


class GmresSolver:
    """https://stackoverflow.com/questions/37962271/whats-wrong-with-my-gmres-implementation"""

    def __init__(self, A, b, maxiter=None, x0=None, tol=1e-10):
        self.A = A
        self.b = b
        self.x0 = x0 if x0 is not None else b
        self.tol = tol

        # check that A is a square matrix
        Ashape = self.A.shape
        assert Ashape[0] == Ashape[1]
        self.N = Ashape[0]

        if maxiter is None:
            maxiter = self.N
        else:
            maxiter = int(maxiter)
        self.maxiter = maxiter

        bshape = self.b.shape
        # assert(len(bshape) == 2 and bshape[1] == 1)

        # initialize helper matricesm H,Q from Arnoldi
        self.Q = np.zeros((self.N, maxiter))
        self.H = np.zeros((maxiter + 1, maxiter))

        # xi unit vector and consecutive rotations
        self.xi = np.zeros((maxiter + 1, 1))
        self.xi[0] = 1.0
        self.cosines = np.zeros((maxiter + 1))
        self.sines = np.zeros((maxiter + 1))
        return

    def solve(self):
        return self.solve2()

    def _main_solve(self):
        """solve the matrix with the desired GMRES algorithm"""
        x = self.x0
        r = self.b - self.A @ x
        beta = np.linalg.norm(r)
        q0 = r / beta
        self.Q[:, 0] = q0[:, 0]

        for k in range(self.maxiter):
            qnext = self.A @ self.Q[:, k]

            # apply Arnoldi's method to update Graham-Schmidt basis
            for j in range(k + 1):
                self.H[j, k] = np.dot(qnext, self.Q[:, j])
                qnext -= self.H[j, k] * self.Q[:, j]
            self.H[k + 1, k] = np.linalg.norm(qnext)
            if self.H[k + 1, k] != 0 and k < self.maxiter - 1:
                qnext /= self.H[k + 1, k]
                self.Q[:, k + 1] = qnext

            # apply previous rotations F_1,...,F_k-1 to kth column of upper Heisenberg
            for i in range(k):
                tempH = self.H[i, k]
                self.H[i, k] = (
                    self.cosines[i] * self.H[i, k] + self.sines[i] * self.H[i + 1, k]
                )
                self.H[i + 1, k] = (
                    -self.sines[i] * tempH + self.cosines[i] * self.H[i + 1, k]
                )

            # calculate entries in F_k or kth rotation matrix to eliminate k+1,k entry of H
            self.cosines[k] = np.abs(self.H[k, k]) / np.sqrt(
                self.H[k, k] ** 2 + self.H[k + 1, k] ** 2
            )
            self.sines[k] = self.cosines[k] * self.H[k + 1, k] / self.H[k, k]

            # apply rotations to eliminate k+1,k entry so that upper Heisenberg
            # matrix is k x k upper triangular matrix
            self.H[k, k] = (
                self.cosines[k] * self.H[k, k] + self.sines[k] * self.H[k + 1, k]
            )
            self.H[k + 1, k] = 0.0

            # apply rotations to the xi unit vector as well
            xi_temp = self.xi[k]
            self.xi[k] = self.cosines[k] * xi_temp
            self.xi[k + 1] = -self.sines[k] * xi_temp

            heisenberg_residual = float(beta * np.abs(self.xi[k + 1]))
            if heisenberg_residual < self.tol:
                break

        # print final residual
        print(f"number iterations = {k+1}")
        print(f"heisenberg residual = {heisenberg_residual}")

        # only once we've minimized residual on the upper Heisenberg do we then compute y and then x
        Hk = self.H[: k + 1, : k + 1]
        xik = self.xi[: k + 1]
        Qk = self.Q[:, : k + 1]

        # solve upper triangular matrix by backsubstitution
        yk = np.zeros((k + 1, 1))
        for row in range(k, -1, -1):
            nright = k - row
            numerator = xik[row]
            for iright in range(nright):
                numerator -= Hk[row, row + iright] * xik[row + iright]
            yk[row, 0] = numerator / Hk[row, row]

        # check that it solves the upper Heisenberg matrix
        heis_res = xik - Hk @ yk
        print(f"heis solve = {np.linalg.norm(heis_res)}")

        xk = self.x0 + Qk @ yk
        return xk

    def solve2(self):
        A = self.A
        x0 = np.reshape(self.x0, newshape=(self.N))
        b = np.reshape(self.b, newshape=(self.N))
        x = self._GMRes(A, b, x0, 0, self.maxiter)
        return x[-1]

    def _GMRes(self, A, b, x0, e, nmax_iter, restart=None):
        r = b - np.asarray(np.dot(A, x0)).reshape(-1)

        x = []
        q = [0] * (nmax_iter)

        x.append(r)

        q[0] = r / np.linalg.norm(r)

        h = np.zeros((nmax_iter + 1, nmax_iter))

        # for k in range(min(nmax_iter, A.shape[0])):
        for k in range(nmax_iter):
            y = np.asarray(np.dot(A, q[k])).reshape(-1)

            for j in range(k + 1):
                h[j, k] = np.dot(q[j], y)
                y = y - h[j, k] * q[j]
            h[k + 1, k] = np.linalg.norm(y)
            if h[k + 1, k] != 0 and k != nmax_iter - 1:
                q[k + 1] = y / h[k + 1, k]

            b = np.zeros(nmax_iter + 1)
            b[0] = np.linalg.norm(r)

            result = np.linalg.lstsq(h, b)[0]
            # print(f"result = {result}")

            x.append(np.dot(np.asarray(q).transpose(), result) + x0)

        return x
