__all__ = ["EulerBernoulliProblem", "EulerBernoulliBC"]

import numpy as np
from aerodesk.linear_algebra import Gmres
from .analysis_function import Function


class EulerBernoulliBC:
    def __init__(self, kind):
        self.kind = kind

    @classmethod
    def pin(cls):
        pass


class EulerBernoulliProblem:
    def __init__(self, elements, bcs, loads):
        """solver for Euler Bernoulli beam problems"""
        self.elements = elements
        self.bcs = bcs
        self.loads = loads

        # global stiffness and force arrays
        self.K = np.zeros((self.ndof, self.ndof))
        self.F = np.zeros((self.ndof, 1))
        self.u = np.zeros((self.ndof, 1))

        # reduced stiffness and forces from BCs
        self.Kred = None
        self.Fred = None
        self.ured = None

        # status booleans
        self._assembled = False
        self._constrained = False

        self._linear_solver = None

        # adjoint states
        self.KT = None
        self.adj_rhs = None
        self.psi = None

        self.rho = 50.0

    @property
    def nelem(self) -> int:
        return len(self.elements)

    @property
    def ndof(self):
        # +4*nelem (since 4 dof/element)
        # -2*nelem (since 2 dof overlap between adj elements)
        # +2 dof (on boundary elements since not cyclic)
        return 2 * self.nelem + 2

    @property
    def variables(self):
        return [elem.thickness_var for elem in self.elements]

    def set_variables(self, xdict):
        """set variables into each element"""
        for varkey in xdict:
            for element in self.elements:
                var = element.thickness_var
                if var.name == varkey:
                    break
            var.value = xdict[varkey]
        return

    def register(self, obj):
        if isinstance(obj, Function):
            self.functions.append(obj)
        return

    def assemble(self):
        """assembly of element stiffness and forces to global stiffness and force arrays"""
        for ielem, element in enumerate(self.elements):
            offset = 2 * ielem

            # distribute loads to this element from global forces
            scale_left = 0.5 if ielem > 0 else 1.0
            scale_right = 0.5 if ielem < self.nelem - 1 else 1.0
            Q1 = scale_left * self.loads[2 * ielem]
            Q2 = scale_left * self.loads[2 * ielem + 1]
            Q3 = scale_right * self.loads[2 * ielem + 2]
            Q4 = scale_right * self.loads[2 * ielem + 3]
            element.set_loads(Q1, Q2, Q3, Q4)

            # add element stiffness and force vectors into global stiffness and force vector
            self.K[offset : offset + 4, offset : offset + 4] += element.stiffness_matrix
            self.F[offset : offset + 4] += element.force_vector
        self._assembled = True
        return

    def apply_bcs(self, bcs=None):
        """apply the boundary conditions to the global system otherwise singular"""
        if bcs is not None:
            self.bcs = bcs
        nred_dof = len(self.reduced_dofs)
        self.Kred = np.zeros((nred_dof, nred_dof))
        for i, ired in enumerate(self.reduced_dofs):
            for j, jred in enumerate(self.reduced_dofs):
                self.Kred[i, j] = self.K[ired, jred]
        self.Fred = self.F[self.reduced_dofs]
        self._constrained = True
        pass

    def solve_forward(self):
        """solve the system, uses GMRES if reduced solution vector not given"""
        if not (self._assembled):
            self.assemble()
        if not (self._constrained):
            self.apply_bcs()

        self._linear_solver = Gmres.solve(A=self.Kred, b=self.Fred)
        self.ured = self.linear_solver.x
        print(f"Euler-Bernoulli forward solve residual = {self.linear_solver.residual}")

        # write reduced displacements back into full displacements
        self.u[self.reduced_dofs] = self.ured
        for ielem, elem in enumerate(self.elements):
            elem.set_displacements(self.u[2 * ielem : 2 * ielem + 4, 0])

        return self.ured

    def solve_adjoint(self):
        """solve the adjoint system, using GMRES if solution is not evaluated elsewhere"""
        # transpose the reduced stiffness matrix
        self.KT = np.transpose(self.Kred)

        # adjoint RHS = -df/du, since only one output function for now
        self.adj_rhs = -self.dstress_du()

        self._adjoint_solver = Gmres.solve(A=self.KT, b=self.adj_rhs)
        self.adj_rhs = self._adjoint_solver.x
        print(
            f"Euler-Bernoulli adjoint solve residual = {self._adjoint_solver.residual}"
        )
        return

    def mass(self):
        tot_mass = 0.0
        for elem in self.elements:
            tot_mass += elem.mass
        return tot_mass

    def dmass_dh(self, var_name):
        """compute dmass/dthickness"""
        for elem in self.elements:
            if elem.thickness_var.name == var_name:
                return elem.dmassdh
        return None

    def stress(self, rho=50.0):
        """ks max stress"""
        inner_sum = 0.0
        for elem in self.elements:
            inner_sum += np.exp(rho * elem.stress)
        return 1.0 / rho * np.log(inner_sum)

    def dstress_du(self, rho=50.0):
        """dstress/d(displacement) used for adjoint"""
        gradient = np.zeros((self.ndof))
        inner_sum = 0.0
        for elem in self.elements:
            inner_sum += np.exp(rho * elem.stress)
        for ielem, elem in enumerate(self.elements):
            dks_dstress = np.exp(rho * elem.stress) / inner_sum
            gradient[2 * ielem : 2 * ielem + 4] = dks_dstress * elem.dstress_du
        return gradient

    def dstress_dh(self, rho=50.0):
        """gradient of ks max stress"""
        gradient = {}
        for ielem, elem in enumerate(self.elements):
            var = elem.thickness_var
            dKdh_u = elem.dKdh @ elem.u
            psi_elem = self.psi[2 * ielem : 2 * ielem + 4]
            gradient[var.name] = np.dot(dKdh_u, psi_elem)
        return gradient

    @property
    def full_displacements(self):
        return self.u

    @property
    def displacements(self):
        return self.ured

    @property
    def global_stiffness_matrix(self):
        return self.K

    @property
    def global_force_vector(self):
        return self.F

    @property
    def stiffness_matrix(self):
        return self.Kred

    @property
    def force_vector(self):
        return self.Fred

    @property
    def full_dofs(self):
        return [_ for _ in range(self.ndof)]

    @property
    def reduced_dofs(self):
        return [_ for _ in self.full_dofs if not (_ in self.bcs)]

    @property
    def linear_solver(self):
        return self._linear_solver

    @property
    def residual(self) -> float:
        return self.linear_solver.residual
