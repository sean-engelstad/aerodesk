import unittest, numpy as np
from aerodesk.fea import (
    EulerBernoulliElement,
    EulerBernoulliProblem,
    ThicknessVariable,
    Isotropic,
)

np.random.seed(1234567)
debug = True


class EulerBernoulliCantileverClosedForm(unittest.TestCase):
    def test_tip_load_closed_form(self):
        L = 2
        P = 1000

        # construct an elementf
        aluminum = Isotropic.test_material()
        thick_var = ThicknessVariable(name="elem1", base=1.0, thickness=0.1)
        element = EulerBernoulliElement(
            material=aluminum, thickness_var=thick_var, x=[0, L]
        )

        eb_problem = EulerBernoulliProblem([element], bcs=[0, 1], loads=[0, 0, P, 0])
        eb_problem.solve_forward()

        # first entry of reduced displacements which was 2x1
        w_FEA = eb_problem.displacements[0, 0]

        # closed-form solution for cantilever beam with tip load
        E = aluminum.E
        I = thick_var.bending_inertia
        w_CF = P * L**3 / 3 / E / I
        rel_error = (w_FEA - w_CF) / w_CF

        print("\nTip Load test...")
        print(f"w FEA = {w_FEA}")
        print(f"w closed-form = {w_CF}")
        print(f"rel error = {rel_error}\n")
        self.assertTrue(abs(rel_error) < 1e-9)
        return

    def test_mid_load_closed_form(self):
        L = 2
        P1 = 1000 * np.random.rand()
        # can tack on P2 at tip load since already verified (but is optional due to superposition)
        P2 = 1200 * np.random.rand()

        # construct an elementf
        aluminum = Isotropic.test_material()
        thick_var1 = ThicknessVariable(name="elem1", base=1.0, thickness=0.1)
        thick_var2 = ThicknessVariable(name="elem2", base=1.0, thickness=0.1)

        element1 = EulerBernoulliElement(
            material=aluminum, thickness_var=thick_var1, x=[0, L / 2]
        )
        element2 = EulerBernoulliElement(
            material=aluminum, thickness_var=thick_var2, x=[L / 2, L]
        )

        eb_problem = EulerBernoulliProblem(
            [element1, element2], bcs=[0, 1], loads=[0, 0, P1, 0, P2, 0]
        )
        eb_problem.solve_forward()

        w_FEA = eb_problem.displacements[2, 0]

        # closed-form solution for cantilever beam with tip load
        E = aluminum.E
        I = thick_var1.bending_inertia
        w_CF = (P1 * 5.0 / 48 + P2 / 3.0) * L**3 / E / I
        rel_error = (w_FEA - w_CF) / w_CF

        print("\nTip and Mid Load test...")
        print(f"w FEA = {w_FEA}")
        print(f"w closed-form = {w_CF}")
        print(f"rel error = {rel_error}\n")
        self.assertTrue(abs(rel_error) < 1e-9)
        return


if __name__ == "__main__":
    unittest.main()
