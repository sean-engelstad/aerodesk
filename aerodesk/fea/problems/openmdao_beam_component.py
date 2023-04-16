__all__ = ["EulerBernoulliBeamComponent"]

import openmdao.api as om
import matplotlib.pyplot as plt


class EulerBernoulliBeamComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("beam_problem", types=object)

        self._iterations = 0
        self._history = {}
        self._history["mass"] = []
        self._history["stress"] = []

    def setup(self):
        """declare the inputs and outputs of the component"""
        beam_problem = self.options["beam_problem"]

        # register each of the thickness variables
        for key in beam_problem.var_dict:
            self.add_input(key, beam_problem.var_dict[key])

        # two different functions
        self.add_output("mass")
        self.add_output("stress")

    def compute(self, inputs, outputs):
        """compute the objective functions"""
        beam_problem = self.options["beam_problem"]

        beam_problem.set_variables(inputs)
        beam_problem.solve_forward()

        outputs["mass"] = beam_problem.mass
        outputs["stress"] = beam_problem.stress

        # update function history
        self._history["mass"].append(beam_problem.mass)
        self._history["stress"].append(beam_problem.stress)
        self._iterations += 1

        self._plot_history()

    def setup_partials(self):
        beam_problem = self.options["beam_problem"]
        for func in ["mass", "stress"]:
            for var in beam_problem.var_dict:
                self.declare_partials(func, var)

    def compute_partials(self, inputs, partials):
        beam_problem = self.options["beam_problem"]

        beam_problem.solve_adjoint()

        for var in beam_problem.var_dict:
            partials["mass", var] = beam_problem.dmass_dh[var]
            partials["stress", var] = beam_problem.dstress_dh[var]

    def _plot_history(self):
        plt.figure("")
        num_iterations = self._iterations
        iterations = [_ for _ in range(num_iterations)]
        plt.plot(iterations, self._history["mass"], label="mass")
        plt.plot(iterations, self._history["stress"], label="stress")
        plt.legend()
        plt.savefig("beam_opt_history.png")
        plt.close("all")
