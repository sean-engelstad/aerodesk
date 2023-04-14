__all__ = ["EulerBernoulliElement"]

import numpy as np


class EulerBernoulliElement:
    def __init__(self, material, x=None, thickness_var=None, q0=0.0, k=0.0):
        """
        class to solve Euler Bernoulli beam problems with FEA

        Parameters
        ----------
        E : float
            modulus of elasticity
        I : float
            bending inertia
        x : list[float]
            list of [xL,xR] the two coordinates of the endpoints of the element
        q0: float
            distributed transverse load on this element
        k : float
            elastic foundation modulus, often 0.0
        """
        assert isinstance(x, list) and len(x) == 2
        self.material = material
        self.thickness_var = thickness_var
        self.x = x
        self.q0 = q0
        self.k = k
        self.he = x[1] - x[0]
        self.Q = None  # list of 4 Q1e, Q2e, Q3e, Q4e

        self._variable = None
        self.u = np.zeros((4, 1))

    @property
    def length(self):
        return self.he

    def register(self, variable):
        """register a thickness variable / some other kind of variable in future"""
        self._variable = variable
        return self

    def set_nodes(self, x):
        self.x = x
        return

    def set_loads(self, Q1, Q2, Q3, Q4, q0=None):
        self.Q = np.array(
            [
                [Q1],
                [Q2],
                [Q3],
                [Q4],
            ]
        )
        if q0 is not None:
            self.q0 = q0
        return

    @property
    def inertia(self):
        return self.thickness_var.bending_inertia

    @property
    def stiffness_matrix(self):
        """return an np array of Ke the element stiffness matrix"""
        he = self.he
        return 2 * self.material.E * self.inertia / he**3 * np.array(
            [
                [6, -3 * he, -6, -3 * he],
                [-3 * he, 2 * he**2, 3 * he, he**2],
                [-6, 3 * he, 6, 3 * he],
                [-3 * he, he**2, 3 * he, 2 * he**2],
            ]
        ) + self.k * he / 420 * np.array(
            [
                [156, -22 * he, 54, 13 * he],
                [-22 * he, 4 * he**2, -13 * he, -3 * he**2],
                [54, -13 * he, 156, 22 * he],
                [13 * he, -3 * he**2, 22 * he, 4 * he**2],
            ]
        )

    @property
    def dKdh(self):
        """derivative of stiffness matrix w.r.t. thickness variable"""
        he = self.he
        return (
            2
            * self.material.E
            * self.thickness_var.dIdh
            / he**3
            * np.array(
                [
                    [6, -3 * he, -6, -3 * he],
                    [-3 * he, 2 * he**2, 3 * he, he**2],
                    [-6, 3 * he, 6, 3 * he],
                    [-3 * he, he**2, 3 * he, 2 * he**2],
                ]
            )
        )

    @property
    def mass(self):
        """mass of the element"""
        return self.material.rho * self.thickness_var.area * self.length

    @property
    def dmassdh(self):
        """mass-thickness derivative of the element"""
        return self.material.rho * self.thickness_var.dAdh * self.length

    def set_displacements(self, u):
        """set displacements in from the problem, once it is solved"""
        self.u[:, 0] = u
        return

    @property
    def strain(self):
        """strain of the element"""
        # get every other entry for linear displacements, not rotational
        ulinear = self.u[0::2, 0]
        return np.diff(ulinear) / self.length

    @property
    def dstrain_du(self):
        """d(strain)/d(displacement) partials"""
        return np.array([-1.0, 1.0]) / self.length

    @property
    def stress(self):
        """stress of the element"""
        return self.material.E * self.strain

    @property
    def dstress_du(self):
        """d(stress)/d(displacement) partials"""
        return self.material.E * self.dstrain_du

    @property
    def force_vector(self):
        """element force vector f_e = q_e + Q_e including distributed and end loads"""
        Qvec = np.zeros((4, 1)) if self.Q is None else self.Q
        q0vec = np.array([[6], [-self.he], [6], [self.he]])
        return self.q0 * self.he / 12.0 * q0vec + Qvec
