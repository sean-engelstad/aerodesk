__all__ = ["EulerBernoulliProblem"]

import numpy as np, os
from aerodesk.linear_algebra import Gmres
from .analysis_function import Function


class EulerBernoulliProblem:
    SOLVERS = ["gmres", "numnpy"]

    def __init__(
        self, elements, bcs, loads, complex=False, solver="gmres", rho=5.0, path=None
    ):
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
        self._solver = solver

        # adjoint states
        self.KT = None
        self.adj_rhs = None
        self.psi = None

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

    @property
    def var_names(self):
        return [var.name for var in self.variables]

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
        # zero out and reset the matrix
        self.K[:, :] = 0.0
        self.F[:, :] = 0.0

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

        if self._solver == "gmres":
            self._linear_solver = Gmres.solve(
                A=self.Kred, b=self.Fred, complex=self.complex
            )
            self.ured = self.linear_solver.x
            print(
                f"Euler-Bernoulli forward solve residual = {self.linear_solver.residual}"
            )
        elif self._solver == "numpy":
            self.ured = np.linalg.solve(self.Kred, self.Fred)

        # write reduced displacements back into full displacements
        self.u[self.reduced_dofs] = self.ured
        for ielem, elem in enumerate(self.elements):
            elem.set_displacements(u=self.u[2 * ielem : 2 * ielem + 4, 0])

        return self.ured

    def solve_adjoint(self):
        """solve the adjoint system, using GMRES if solution is not evaluated elsewhere"""
        # transpose the reduced stiffness matrix
        self.KT = np.transpose(self.K)
        # bc rows get zeroed out and replaced with identity
        for bc in self.bcs:
            self.KT[bc, :] = 0.0
            self.KT[bc, bc] = 1.0

        # adjoint RHS = -df/du, since only one output function for now
        self.adj_rhs = -self.dstress_du

        if self._solver == "gmres":
            self._adjoint_solver = Gmres.solve(
                A=self.KT, b=self.adj_rhs, complex=self.complex
            )
            self.psi = self._adjoint_solver.x
            print(
                f"Euler-Bernoulli adjoint solve residual = {self._adjoint_solver.residual}"
            )
        elif self._solver == "numpy":
            self.psi = np.linalg.solve(self.KT, self.adj_rhs)
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
            offset = 2 * ielem
            gradient[offset : offset + 4, 0] += dks_dstress * elem.dstress_du
        return gradient

    @property
    def dstress_dh(self):
        """gradient of ks max stress"""
        gradient = {}
        for ielem, elem in enumerate(self.elements):
            var = elem.thickness_var
            dKdhu_global = np.zeros((self.ndof, 1), dtype=self.dtype)
            dKdh_u = elem.dKdh @ elem.u
            offset = 2 * ielem
            dKdhu_global[offset : offset + 4, :] = dKdh_u
            gradient[var.name] = dKdhu_global.T @ self.psi
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
    def nred_dof(self):
        return len(self.reduced_dofs)

    @property
    def linear_solver(self):
        return self._linear_solver

    @property
    def residual(self) -> float:
        return self.linear_solver.residual

    def write_vtk(self, path=None, prefix="eb_beam", index=0):
        """write a vtk file for the beam"""
        nnodes = self.nelem + 1
        npoints = 2 * nnodes

        if path is None:
            path = os.getcwd()

        filename = f"{prefix}_{index}.vtk"
        filepath = os.path.join(path, filename)
        hdl = open(filepath, "w")
        hdl.write(f"# vtk DataFile Version 2.0\n")
        hdl.write(f"Euler Bernoulli Beam\n")
        hdl.write(f"ASCII\n")
        hdl.write(f"DATASET UNSTRUCTURED_GRID\n")
        hdl.write(f"POINTS {npoints} double64\n")

        # write all the point coordinates
        # write initial two points on left of first element
        elements = self.elements
        elem0 = elements[0]
        xleft = float(elem0.x[0])
        h = float(elem0.thickness)
        hdl.write(f"{xleft} {h/2.0} 0.0\n")
        hdl.write(f"{xleft} {-h/2.0} 0.0\n")

        # loop over remaining points, right side of each element
        for elem in self.elements:
            xright = float(elem.x[1])
            h = float(elem.thickness)
            hdl.write(f"{xright} {h/2.0} 0.0\n")
            hdl.write(f"{xright} {-h/2.0} 0.0\n")

        # write out each of the cells
        ncells = self.nelem
        hdl.write(f"CELLS {ncells} {5*ncells}\n")
        for ielem in range(self.nelem):
            # {x},{y}pt L or right each
            BLpt = 2 * ielem
            BRpt = 2 * ielem + 2
            URpt = 2 * ielem + 3
            ULpt = 2 * ielem + 1
            hdl.write(f"4 {BLpt} {BRpt} {URpt} {ULpt}\n")

        # write out cell types
        hdl.write(f"CELL_TYPES {ncells}\n")
        for ielem in range(self.nelem):
            hdl.write("9\n")

        # vector data u,v,w displacements
        # plotting v here, theory is w, and we say u in the above code
        hdl.write(f"POINT_DATA {npoints}\n")
        hdl.write(f"VECTORS DISP double64\n")
        # left node of first element
        v = float(self.u[0].real)
        hdl.write(f"0.0 {v} 0.0\n")
        hdl.write(f"0.0 {v} 0.0\n")
        for ielem, elem in enumerate(self.elements):
            v = float(elem.u[2].real)
            hdl.write(f"0.0 {v} 0.0\n")
            hdl.write(f"0.0 {v} 0.0\n")

        hdl.close()
        return
