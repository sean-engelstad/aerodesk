import numpy as np, openmdao.api as om
from aerodesk import (
    EulerBernoulliBeamComponent,
    EulerBernoulliElement,
    EulerBernoulliProblem,
    Isotropic,
    ThicknessVariable,
)

# setup the euler bernoulli beam element
L = 10
P1 = 10 * np.random.rand()
P2 = 12 * np.random.rand()

# construct an element
aluminum = Isotropic.test_material()
N = 100
elements = []
loads = [0, 0]
for ielem in range(N):
    thickness = 0.8 + 0.01 * ielem
    thick_var = ThicknessVariable(name=f"elem{ielem}", base=2.0, thickness=thickness)
    xleft = ielem * 1.0 / N
    xright = xleft + 1.0 / N
    xleft *= L
    xright *= L
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
    elements, bcs=[0, 1], loads=loads, complex=True, solver="numpy", rho=20.0
)

# check initial mass and stress values
eb_problem.solve_forward()
print(f"mass = {eb_problem.mass}")
print(f"stress = {eb_problem.stress}")

# eb_problem.write_vtk()
# import sys
# sys.exit()

# setup the OpenMDAO Problem object
om_problem = om.Problem()

# Create the OpenMDAO component
beam_system = EulerBernoulliBeamComponent(beam_problem=eb_problem, write_vtk=True)
om_problem.model.add_subsystem("ebSystem", beam_system)

# setup the optimizer settings
om_problem.driver = om.ScipyOptimizeDriver()
om_problem.driver.options["optimizer"] = "SLSQP"
om_problem.driver.options["tol"] = 1.0e-9
om_problem.driver.options["disp"] = True

# add design variables to the model
for thick_dv in eb_problem.var_names:
    om_problem.model.add_design_var(f"ebSystem.{thick_dv}", lower=0.001, upper=10.0)

# add objectives & constraints to the model
om_problem.model.add_objective("ebSystem.mass")
om_problem.model.add_constraint("ebSystem.stress", upper=0.267)

# Start the optimization
print("\n==> Starting optimization...")
om_problem.setup()
om_problem.run_driver()
om_problem.cleanup()

# report the final optimal design
for thick_dv in eb_problem.var_names:
    opt_value = om_problem.get_val(f"ebSystem.{thick_dv}")
    print(f"\t{thick_dv} = {opt_value}")

# write the file to vtk
eb_problem.write_vtk()
