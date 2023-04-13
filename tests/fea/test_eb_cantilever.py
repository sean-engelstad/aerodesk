import unittest, numpy as np
from aerodesk.fea import EulerBernoulliElement, EulerBernoulliProblem

np.random.seed(1234567)
debug = False


class EulerBernoulliCantilever(unittest.TestCase):
    def test_tip_load(self):
        E = 1e9
        L = 2
        P = 1000

        # construct an elementf
        I = EulerBernoulliElement.bending_inertia(base=1.0, height=0.1)
        element = EulerBernoulliElement(E=E, I=I, x=[0, L])

        eb_problem = EulerBernoulliProblem([element], bcs=[0, 1], loads=[0, 0, P, 0])
        eb_problem.solve()

        if debug:
            r = (
                eb_problem.force_vector
                - eb_problem.stiffness_matrix @ eb_problem.displacements
            )
            print(f"residual vec = {r}")

        # first entry of reduced displacements which was 2x1
        w_FEA = eb_problem.displacements[0, 0]

        # closed-form solution for cantilever beam with tip load
        w_CF = P * L**3 / 3 / E / I
        rel_error = (w_FEA - w_CF) / w_CF

        print("\nTip Load test...")
        print(f"w FEA = {w_FEA}")
        print(f"w closed-form = {w_CF}")
        print(f"rel error = {rel_error}\n")
        self.assertTrue(abs(rel_error) < 1e-9)

    def test_tip_and_mid_load(self):
        E = 1e9
        L = 2
        P1 = 1000 * np.random.rand()
        P2 = 1200 * np.random.rand()

        # construct an elementf
        I = EulerBernoulliElement.bending_inertia(base=1.0, height=0.1)
        element1 = EulerBernoulliElement(E=E, I=I, x=[0, L / 2])
        element2 = EulerBernoulliElement(E=E, I=I, x=[L / 2, L])

        eb_problem = EulerBernoulliProblem(
            [element1, element2], bcs=[0, 1], loads=[0, 0, P1, 0, P2, 0]
        )
        eb_problem.solve()

        if debug:
            # check determinant of stiffness matrix
            K = eb_problem.stiffness_matrix
            Kdet = np.linalg.det(K)
            print(f"Kfull = {eb_problem.K}")
            print(f"Kred = {K}")
            print(f"K det = {Kdet}")

            r = (
                eb_problem.force_vector
                - eb_problem.stiffness_matrix @ eb_problem.displacements
            )
            print(f"residual vec = {r}")

            print(f"eb displacements 2 = {eb_problem.displacements}")

        w_FEA = eb_problem.displacements[2, 0]

        # closed-form solution for cantilever beam with tip load
        w_CF = (P1 * 5.0 / 48 + P2 / 3.0) * L**3 / E / I
        rel_error = (w_FEA - w_CF) / w_CF

        print("\nTip and Mid Load test...")
        print(f"w FEA = {w_FEA}")
        print(f"w closed-form = {w_CF}")
        print(f"rel error = {rel_error}\n")
        self.assertTrue(abs(rel_error) < 1e-9)


if __name__ == "__main__":
    unittest.main()
