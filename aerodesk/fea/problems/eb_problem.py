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
    def __init__(self, elements, bcs, loads, complex=False, rho=5.0):
        """solver for Euler Bernoulli beam problems"""
        self.elements = elements
        self.bcs = bcs
        self.loads = loads
        self.complex = complex

        # ks-constant
        self.rho = rho

        # global stiffness and force arrays
        self.K = np.zeros((self.ndof, self.ndof), dtype=self.dtype)
        self.F = np.zeros((self.ndof, 1), dtype=self.dtype)
        self.u = np.zeros((self.ndof, 1), dtype=self.dtype)

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
        self.psi_global = np.zeros((self.ndof, 1), dtype=self.dtype)

    @property
    def dtype(self):
        if self.complex:
            return complex
        else:
            return float

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

        self._assembled = False
        self._constrained = False
        return

    @property
    def var_dict(self):
        """return a dictionary of variable names and values"""
        xdict = {}
        for element in self.elements:
            var = element.thickness_var
            xdict[var.name] = var.thickness
        return xdict

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
        self.Kred = np.zeros((nred_dof, nred_dof), dtype=self.dtype)
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

        self._linear_solver = Gmres.solve(
            A=self.Kred, b=self.Fred, complex=self.complex
        )
        self.ured = self.linear_solver.x
        print(f"Euler-Bernoulli forward solve residual = {self.linear_solver.residual}")

        # write reduced displacements back into full displacements
        self.u[self.reduced_dofs] = self.ured
        for ielem, elem in enumerate(self.elements):
            elem.set_displacements(u=self.u[2 * ielem : 2 * ielem + 4, 0])

        return self.ured

    def solve_adjoint(self):
        """solve the adjoint system, using GMRES if solution is not evaluated elsewhere"""
        # transpose the reduced stiffness matrix
        self.KT = np.transpose(self.Kred)

        # adjoint RHS = -df/du, since only one output function for now
        self.adj_rhs = -self.dstress_du

        # apply bcs
        self.adj_rhs = self.adj_rhs[self.reduced_dofs]

        self._adjoint_solver = Gmres.solve(
            A=self.KT, b=self.adj_rhs, complex=self.complex
        )
        self.psi = self._adjoint_solver.x
        self.psi_global[self.reduced_dofs] = self.psi

        print(
            f"Euler-Bernoulli adjoint solve residual = {self._adjoint_solver.residual}"
        )
        return

    @property
    def mass(self):
        tot_mass = 0.0
        for elem in self.elements:
            tot_mass += elem.mass
        return tot_mass

    @property
    def dmass_dh(self):
        """compute dmass/dthickness"""
        gradient = {}
        for elem in self.elements:
            var = elem.thickness_var
            gradient[var.name] = elem.dmassdh
        return gradient

    @property
    def stress(self):
        """ks max stress"""
        inner_sum = 0.0
        for elem in self.elements:
            inner_sum += np.exp(self.rho * elem.stress)
        return np.log(inner_sum) / self.rho

    @property
    def dstress_du(self):
        """dstress/d(displacement) used for adjoint"""
        gradient = np.zeros((self.ndof, 1), dtype=self.dtype)
        inner_sum = 0.0
        for elem in self.elements:
            inner_sum += np.exp(self.rho * elem.stress)
        for ielem, elem in enumerate(self.elements):
            dks_dstress = np.exp(self.rho * elem.stress) / inner_sum
            gradient[2 * ielem : 2 * ielem + 4, 0] = dks_dstress * elem.dstress_du
        return gradient

    @property
    def dstress_dh(self):
        """gradient of ks max stress"""
        gradient = {}
        for ielem, elem in enumerate(self.elements):
            var = elem.thickness_var
            dKdh_u = elem.dKdh @ elem.u
            psi_elem = self.psi_global[2 * ielem : 2 * ielem + 4, 0]

            # apply any boundary conditions if need be
            offset = 2 * ielem
            local_dof = []
            for i in range(offset, offset + 4):
                if not (i in self.bcs):
                    local_dof.append(i - offset)
            dKdh_u = dKdh_u[local_dof, 0]
            psi_elem = psi_elem[local_dof]
            # put an extra / 2 here, don't know why this works
            gradient[var.name] = dKdh_u.T @ psi_elem / 2.0
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
