__all__ = ["Gmres"]

import numpy as np


class Gmres:
    """
    Solves linear systems of the form Ax = b using the GMRES Algorithm
    GMRES stands for Generalized Minimal Residual
    """

    def __init__(
        self, A, b, maxiter=None, x0=None, tol=1e-10, debug=False, complex=False
    ):
        self.A = A
        self.b = b
        self.x0 = x0 if x0 is not None else b
        self.tol = tol
        self.debug = debug
        self.complex = complex

        # check that A is a square matrix
        Ashape = self.A.shape
        assert Ashape[0] == Ashape[1]
        self.N = Ashape[0]

        # if maxiter is None:
        #     maxiter = self.N
        # else:
        #     maxiter = int(maxiter)
        #     if maxiter > self.N:
        #         maxiter = self.N
        maxiter = int(maxiter)
        self.maxiter = maxiter

        bshape = self.b.shape
        assert len(bshape) == 2 and bshape[1] == 1

        # initialize helper matricesm H,Q from Arnoldi
        self.Q = np.zeros((self.N, maxiter + 1), dtype=self.dtype)
        self.H = np.zeros((maxiter + 1, maxiter), dtype=self.dtype)

        # xi unit vector and consecutive rotations
        self.xi = np.zeros((maxiter + 1, 1), dtype=self.dtype)
        self.xi[0] = 1.0
        self.cosines = np.zeros((maxiter), dtype=self.dtype)
        self.sines = np.zeros((maxiter), dtype=self.dtype)

        self._x = None
        return

    @property
    def dtype(self):
        if self.complex:
            return np.longcomplex
        else:
            return np.longdouble

    def jacobi_precondition(self):
        # extract the diagonal of the matrix
        diagonal = np.diag(self.A)
        inv_diagonal = 1.0 / diagonal
        Dinv = np.diag(inv_diagonal)
        self.A = Dinv @ self.A
        self.b = Dinv @ self.b
        # f_diagonal = np.diag(self.A)
        return self

    @property
    def x(self):
        return self._x

    @property
    def residual(self):
        if self._x is None:
            return None
        else:
            r = self.b - self.A @ self.x
            return np.linalg.norm(r)

    @classmethod
    def solve(cls, A, b, tol=1e-10, maxiter=1e3, x0=None, debug=False, complex=False):
        return (
            cls(A, b, tol=tol, maxiter=maxiter, x0=x0, debug=debug, complex=complex)
            .jacobi_precondition()
            .solve_system()
        )

    def solve_system(self):
        x = self.x0
        r = self.b - self.A @ x
        beta = np.linalg.norm(r)
        q = r / beta
        self.Q[:, 0] = q[:, 0]

        # Arnoldi's iteration
        for k in range(self.maxiter):
            qbar = self.A @ q
            for i in range(k + 1):
                self.H[i, k] = np.conjugate(np.conjugate(qbar).T @ self.Q[:, i])
            for i in range(k + 1):
                qbar[:, 0] -= self.H[i, k] * self.Q[:, i]
            self.H[k + 1, k] = np.linalg.norm(qbar)
            if abs(self.H[k + 1, k]) < 1e-12:
                q = 0 * qbar
            else:
                q = qbar / self.H[k + 1, k]
            self.Q[:, k + 1] = q[:, 0]

            # apply previous Gibben's rotations to H
            for i in range(k):
                temp = self.H[i, k]
                ci = self.cosines[i]
                si = self.sines[i]
                self.H[i, k] = ci * temp + si * self.H[i + 1, k]
                self.H[i + 1, k] = -np.conjugate(si) * temp + ci * self.H[i + 1, k]

            # compute kth Gibben's rotation
            d = self.H[k, k]
            h = self.H[k + 1, k]
            if np.abs(d) == 0.0:
                self.cosines[k] = 0.0
                self.sines[k] = 1.0
            else:
                self.cosines[k] = np.abs(d) / np.sqrt(np.abs(d) ** 2 + np.abs(h) ** 2)
                conj_sine = self.cosines[k] * h / d
                self.sines[k] = np.conjugate(conj_sine)

            # perform kth Gibben's rotation to H and xi
            temp = self.H[k, k]
            ck = self.cosines[k]
            sk = self.sines[k]
            self.H[k, k] = ck * temp + sk * self.H[k + 1, k]
            self.H[k + 1, k] = -np.conjugate(sk) * temp + ck * self.H[k + 1, k]

            temp = self.xi[k, 0]
            self.xi[k, 0] = ck * temp + sk * self.xi[k + 1, 0]
            self.xi[k + 1, 0] = -np.conjugate(sk) * temp + ck * self.xi[k + 1, 0]

            rtol = np.abs(beta * self.xi[k + 1, 0])
            if rtol < self.tol:
                break

        if self.debug:
            print(f"Q = {self.Q}")
            print(f"H = {self.H}")
            print(f"cosines = {self.cosines}")
            print(f"sines = {self.sines}")
            print(f"xi = {self.xi}")

        # solve the upper triangular system
        # RHS = beta*(F*xi)_kx1
        by = beta * self.xi[: k + 1, :]

        # solve the upper triangular system
        y = np.zeros((k + 1, 1), dtype=self.dtype)
        for i in range(k, -1, -1):
            nright = k - i
            numerator = by[i, 0]
            for iright in range(1, nright + 1):
                numerator -= self.H[i, i + iright] * y[i + iright, 0]
            y[i, 0] = numerator / self.H[i, i]

        # check the upper triangular system was solved
        if self.debug:
            R = self.H[: k + 1, : k + 1]
            upper_triangular_diff = R @ y - by
            print(f"R = {R}")
            print(f"by = {by}")
            print(f"y = {y}")
            print(f"upper triangular system check = {upper_triangular_diff}")

        # compute solution xk = x0 + Qk * yk
        self._x = self.x0 + self.Q[:, : k + 1] @ y
        return self
