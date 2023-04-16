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

    def test_dstress_du_beam(self):
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
        self.assertTrue(abs(dstress_du_rel_error) < 1e-8)

        return

    def test_adjoint_mid_load(self):
        L = 2
        P1 = 1000 * np.random.rand()
        P2 = 1200 * np.random.rand()

        # construct an element
        aluminum = Isotropic.test_material()
        N = 10
        elements = []
        loads = []
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

        print(f"loads = {loads}")

        eb_problem = EulerBernoulliProblem(
            elements, bcs=[0, 1], loads=loads, complex=True
        )

        """perform complex step over mass and stress functions"""
        # random input contravariant tensor d(variable)/ds
        x0dict = eb_problem.var_dict
        dvar_ds = {key: np.random.rand() for key in x0dict}

        h = 1e-30  # step size in complex mode

        # adjoint method
        eb_problem.solve_forward()
        eb_problem.solve_adjoint()

        dmass_ds_adj = 0.0
        dstress_ds_adj = 0.0
        for varname in eb_problem.var_dict:
            dmass_ds_adj += eb_problem.dmass_dh[varname] * dvar_ds[varname]
            dstress_ds_adj += eb_problem.dstress_dh[varname] * dvar_ds[varname]

        # complex-step method
        # does my GMRES solver work with complex numbers
        xdict = {key: x0dict[key] + dvar_ds[key] * h * 1j for key in x0dict}
        eb_problem.set_variables(xdict)

        eb_problem.solve_forward()

        dmass_ds_cmplx = np.imag(eb_problem.mass) / h
        dstress_ds_cmplx = np.imag(eb_problem.stress) / h

        # complex-step result for mass function
        mass_rel_error = (dmass_ds_adj - dmass_ds_cmplx) / dmass_ds_cmplx
        mass_rel_error = np.abs(complex(mass_rel_error))
        print(f"dmass/ds adjoint = {dmass_ds_adj}")
        print(f"dmass/ds complex step = {dmass_ds_cmplx}")
        print(f"mass relative error = {mass_rel_error}")

        stress_rel_error = (dstress_ds_adj - dstress_ds_cmplx) / dstress_ds_cmplx
        stress_rel_error = np.abs(complex(stress_rel_error))
        print(f"dstress/ds adjoint = {dstress_ds_adj}")
        print(f"dstress/ds complex step = {dstress_ds_cmplx}")
        print(f"stress relative error = {stress_rel_error}")

        tol = 1e-6
        self.assertTrue(abs(mass_rel_error) < tol)
        self.assertTrue(abs(stress_rel_error) < tol)
        return


if __name__ == "__main__":
    unittest.main()
