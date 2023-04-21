__all__ = ["EulerBernoulliBeamComponent"]

import openmdao.api as om, matplotlib.pyplot as plt
import os


class EulerBernoulliBeamComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("beam_problem", types=object)
        self.options.declare("write_vtk", types=bool)

        self._iterations = 0
        self._history = {}
        self._history["mass"] = []
        self._history["stress"] = []

        # make a results folder
        self.vtk_folder = os.path.join(os.getcwd(), "vtk")
        if not (os.path.exists(self.vtk_folder)):
            os.mkdir(self.vtk_folder)

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
        write_vtk = self.options["write_vtk"]

        beam_problem.set_variables(inputs)
        beam_problem.solve_forward()

        mass = beam_problem.mass.real
        stress = beam_problem.stress.real

        outputs["mass"] = mass
        outputs["stress"] = stress

        # update function history
        self._history["mass"].append(mass)
        self._history["stress"].append(stress)
        self._iterations += 1

        # write to a vtk if desired
        beam_problem.write_vtk(path=self.vtk_folder, index=self._iterations)

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
            partials["mass", var] = beam_problem.dmass_dh[var].real
            partials["stress", var] = beam_problem.dstress_dh[var].real

    def _plot_history(self):
        plt.figure("")
        num_iterations = self._iterations
        iterations = [_ for _ in range(num_iterations)]
        plt.plot(iterations, self._history["mass"], label="mass")
        plt.plot(iterations, self._history["stress"], label="stress")
        plt.legend()
        plt.savefig("beam_opt_history.png")
        plt.close("all")
