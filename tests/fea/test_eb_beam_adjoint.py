import unittest, numpy as np
from aerodesk.fea import (
    EulerBernoulliElement,
    EulerBernoulliProblem,
    ThicknessVariable,
    Isotropic,
)

np.random.seed(1234567)
debug = True

# solver = "gmres"
solver = "numpy"


class EulerBernoulliCantileverAdjoint(unittest.TestCase):
    def test_adjoint_mid_load(self):
        L = 2
        P1 = 1000 * np.random.rand()
        P2 = 1200 * np.random.rand()

        # construct an element
        aluminum = Isotropic.test_material()
        N = 10
        elements = []
        loads = [0, 0]
        for ielem in range(N):
            thick_var = ThicknessVariable(
                name=f"elem{ielem}", base=1.0, thickness=0.1  # + ielem * 0.01
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
            elements, bcs=[0, 1], loads=loads, complex=True, solver=solver, tol=1e-20
        )

        """perform complex step over mass and stress functions"""
        # random input contravariant tensor d(variable)/ds
        x0dict = eb_problem.var_dict
        dvar_ds = {key: np.random.rand() for key in x0dict}

        # adjoint method
        eb_problem.solve_forward()
        eb_problem.solve_adjoint()

        # print(f"eb_problem K = {eb_problem.K}", flush=True)

        dmass_ds_adj = 0.0
        dstress_ds_adj = 0.0
        for varname in eb_problem.var_dict:
            dmass_ds_adj += eb_problem.dmass_dh[varname]  # * dvar_ds[varname]
            dstress_ds_adj += eb_problem.dstress_dh[varname]  # * dvar_ds[varname]

        # # complex-step method
        # # does my GMRES solver work with complex numbers
        h = 1e-30  # step size in complex mode
        xdict = {key: x0dict[key] + 1.0 * h * 1j for key in x0dict}
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
