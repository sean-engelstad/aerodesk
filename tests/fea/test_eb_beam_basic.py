import unittest, numpy as np
from aerodesk.fea import (
    EulerBernoulliElement,
    EulerBernoulliProblem,
    ThicknessVariable,
    Isotropic,
)

np.random.seed(1234567)
debug = True


class EulerBernoulliCantileverAdjoint(unittest.TestCase):
    def test_inertia_jacobian(self):
        rel_error = ThicknessVariable.test_inertia_jacobian(base=1.0, height=0.5)
        print(f"rel error inertia jacobian = {rel_error}")
        self.assertTrue(abs(rel_error) < 1.0e-8)
        return

    def test_dKdh_element(self):
        # random output contravariant tensor
        dLdK = np.random.rand(4, 4)

        aluminum = Isotropic.test_material()
        thick_var1 = ThicknessVariable(name="elem1", base=1.0, thickness=0.1)
        # thick_var2 = ThicknessVariable(name="elem2", base=1.0, thickness=0.1)

        L = 2.0
        element1 = EulerBernoulliElement(aluminum, thick_var1, complex=True, x=[0, L])

        # compute dKdh naturally / adjoint kind of
        dLdh_adj = 0.0
        dKdh = element1.dKdh
        for i in range(4):
            for j in range(4):
                dLdh_adj += dLdK[i, j] * dKdh[i, j]

        # compute dLdh using complex step
        h = 1e-30
        element1.thickness_var.value = 0.1 + 1j * h
        dKdh2 = np.imag(element1.stiffness_matrix) / h

        dLdh_cmplx = 0.0
        for i in range(4):
            for j in range(4):
                dLdh_cmplx += dLdK[i, j] * dKdh2[i, j]

        rel_error = (dLdh_adj - dLdh_cmplx) / dLdh_cmplx

        print(f"dLdh adjoint = {dLdh_adj}")
        print(f"dLdh complex = {dLdh_cmplx}")
        print(f"rel error dLdh = {rel_error}")
        self.assertTrue(abs(rel_error) < 1.0e-8)
        return

    def test_dstress_du_element(self):
        # random output contravariant tensor
        randu = np.random.rand(4)
        duds = np.random.rand(4)

        aluminum = Isotropic.test_material()
        thick_var1 = ThicknessVariable(name="elem1", base=1.0, thickness=0.1)

        L = 2.0
        element1 = EulerBernoulliElement(aluminum, thick_var1, complex=True, x=[0, L])

        # adjoint method
        element1.set_displacements(u=randu)
        dstress_du = element1.dstress_du
        dstress_ds_adj = np.sum(dstress_du * duds)

        # complex step
        h = 1e-30
        element1.set_displacements(u=randu + 1j * h * duds)
        dstress_ds_cmplx = np.imag(element1.stress) / h

        rel_error_dstress_du = (dstress_ds_adj - dstress_ds_cmplx) / dstress_ds_cmplx
        print(f"dstress/du adjoint = {dstress_ds_adj}")
        print(f"dstress/du complex = {dstress_ds_cmplx}")
        print(f"dstress/du rel error = {rel_error_dstress_du}")
        self.assertTrue(abs(rel_error_dstress_du) < 1e-8)
        return

    def test_dstress_du_element(self):
        L = 2
        P1 = 1000 * np.random.rand()

        # construct an element and beam problem (1 element test)
        aluminum = Isotropic.test_material()
        thick_var1 = ThicknessVariable(name="elem1", base=1.0, thickness=0.1)
        element1 = EulerBernoulliElement(aluminum, thick_var1, complex=True, x=[0, L])

        eb_problem = EulerBernoulliProblem(
            [element1], bcs=[0, 1], loads=[0, 0, P1, 0], complex=True
        )

        # random contravariant tensor du/ds
        du_ds = np.random.rand(4)
        randu = np.random.rand(4)

        # adjoint dstress/du
        element1.u[:, 0] = randu
        dstress_du = eb_problem.dstress_du[:, 0]
        dstress_ds_adj = np.sum(dstress_du * du_ds)

        # complex step
        h = 1e-30
        element1.u[:, 0] = randu + 1j * h * du_ds
        dstress_ds_cmplx = np.imag(eb_problem.stress) / h

        dstress_du_rel_error = (dstress_ds_adj - dstress_ds_cmplx) / dstress_ds_cmplx
        print(f"dstress/du beam problem adjoint = {dstress_ds_adj}")
        print(f"dstress/du beam problem cmplx = {dstress_ds_cmplx}")
        print(f"dstress/du beam problem rel error = {dstress_du_rel_error}")
        # self.assertTrue(abs(dstress_du_rel_error) < 1e-8)
        return

    def test_dstress_du_beam(self):
        L = 2
        P1 = 1000 * np.random.rand()
        P2 = 1200 * np.random.rand()

        # construct an element
        aluminum = Isotropic.test_material()
        N = 4
        elements = []
        loads = [0, 0]
        for ielem in range(N):
            thick_var = ThicknessVariable(
                name=f"elem{ielem}", base=1.0, thickness=0.1 + ielem * 0.01
            )
            xleft = ielem * 1.0 / N
            xright = xleft + 1.0 / N
            element = EulerBernoulliElement(
                aluminum, thick_var, complex=True, x=[xleft, xright]
            )
            elements.append(element)
            if ielem == int(N / 2.0):
                loads += [P1, 0]
            elif ielem == N - 1:
                loads += [P2, 0]
            else:
                loads += [0, 0]

        eb_problem = EulerBernoulliProblem(
            elements, bcs=[0, 1], loads=loads, complex=True, rho=1.0
        )

        # get ndof of Kred
        ndof = eb_problem.ndof
        duds = np.random.rand(ndof, 1)

        """perform complex step over mass and stress functions"""
        # random input contravariant tensor d(variable)/ds
        x0dict = eb_problem.var_dict
        dvar_ds = {key: np.random.rand() for key in x0dict}

        h = 1e-30  # step size in complex mode

        # adjoint method
        eb_problem.solve_forward()

        dstress_ds_adj = eb_problem.dstress_du.T @ duds

        # complex-step method
        # does my GMRES solver work with complex numbers
        h = 1e-30
        # eb_problem.u[pert_ind] += 1j * h
        eb_problem.u += 1j * h * duds
        for ielem, elem in enumerate(eb_problem.elements):
            elem.set_displacements(u=eb_problem.u[2 * ielem : 2 * ielem + 4, 0])

        dstress_ds_cmplx = np.imag(eb_problem.stress) / h

        # complex-step result for mass function
        dstress_ds_rel_error = (dstress_ds_adj - dstress_ds_cmplx) / dstress_ds_cmplx
        dstress_ds_rel_error = np.abs(complex(dstress_ds_rel_error))
        print(f"dstress_ds_adj = {dstress_ds_adj}")
        print(f"dstress_ds_cmplx = {dstress_ds_cmplx}")
        print(f"dstress_ds_rel_error = {dstress_ds_rel_error}", flush=True)
        self.assertTrue(abs(dstress_ds_rel_error) < 1e-8)
        return

    def test_dKdh_problem(self):
        L = 2
        P1 = 1000 * np.random.rand()
        P2 = 1200 * np.random.rand()

        # construct an element
        aluminum = Isotropic.test_material()
        N = 4
        elements = []
        loads = [0, 0]
        for ielem in range(N):
            thick_var = ThicknessVariable(
                name=f"elem{ielem}", base=1.0, thickness=0.1 + ielem * 0.01
            )
            xleft = ielem * 1.0 / N
            xright = xleft + 1.0 / N
            element = EulerBernoulliElement(
                aluminum, thick_var, complex=True, x=[xleft, xright]
            )
            elements.append(element)
            if ielem == int(N / 2.0):
                loads += [P1, 0]
            elif ielem == N - 1:
                loads += [P2, 0]
            else:
                loads += [0, 0]

        eb_problem = EulerBernoulliProblem(
            elements, bcs=[0, 1], loads=loads, complex=True
        )

        # get ndof of Kred
        nred_dof = eb_problem.nred_dof
        ndof = eb_problem.ndof
        dLdK = np.random.rand(ndof, ndof)

        """perform complex step over mass and stress functions"""
        # random input contravariant tensor d(variable)/ds
        x0dict = eb_problem.var_dict
        dvar_ds = {key: np.random.rand() for key in x0dict}

        h = 1e-30  # step size in complex mode

        # adjoint method
        eb_problem.solve_forward()
        # eb_problem.solve_adjoint()

        # dKdh for each element
        dLds_adj = 0.0
        for ielem, elem in enumerate(eb_problem.elements):
            dKdh_full = np.zeros((ndof, ndof), dtype=complex)
            offset = 2 * ielem
            dKdh_full[offset : offset + 4, offset : offset + 4] = elem.dKdh
            dLdh = np.sum(dLdK * dKdh_full)
            dLds_adj += dLdh * dvar_ds[elem.name]

        # complex step for each dKdh
        dLds_cmplx = 0.0

        # complex-step method
        # does my GMRES solver work with complex numbers
        xdict = {key: x0dict[key] + dvar_ds[key] * h * 1j for key in x0dict}
        eb_problem.set_variables(xdict)

        eb_problem.solve_forward()

        dKds = np.imag(eb_problem.K) / h
        dLds_cmplx = np.sum(dLdK * dKds)

        # complex-step result for mass function
        dKdh_rel_error = (dLds_adj - dLds_cmplx) / dLds_cmplx
        dKdh_rel_error = np.abs(complex(dKdh_rel_error))
        print(f"dKdh adjoint = {dLds_adj}")
        print(f"dKdh complex step = {dLds_cmplx}")
        print(f"dKdh relative error = {dKdh_rel_error}")
        self.assertTrue(abs(dKdh_rel_error) < 1e-8)
        return


if __name__ == "__main__":
    unittest.main()
