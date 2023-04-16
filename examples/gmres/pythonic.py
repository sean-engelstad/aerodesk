from aerodesk import Gmres
import numpy as np

N = 2000
A = np.random.rand(N, N)
b = np.random.rand(N, 1)
if N < 1000:
    print(f"A = {A}")
    print(f"b = {b}")

solver = Gmres.solve(A, b, tol=1e-18, maxiter=1e5)
if N < 1000:
    print(f"x = {solver.x}")
print(f"resid = {solver.residual}")
