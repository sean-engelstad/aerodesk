"""
Sean Engelstad, April 15
Debug GMRES solver for complex matrices / vectors
"""

import numpy as np

np.random.seed(12345)

# generate a small 4x4 matrix and 4x1 vector system A x = b
# so the problem is manageable for a human
A = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
b = np.random.rand(4, 1) + 1j * np.random.rand(4, 1)

print(f"A = {A}")
print(f"b = {b}")

# initial guess x0 = b in this case
x0 = b
x = x0
r = b - A @ x
rnorm = np.linalg.norm(r)
rnorm2 = np.sqrt(np.conjugate(r).T @ r)
print(f"rnorm1 = {rnorm}, rnorm2 = {rnorm2}")
q1 = r / rnorm2
q1norm = np.sqrt(np.conjugate(q1).T @ q1)
print(f"q1 = {q1}")
print(f"q1 norm = {q1norm}")

# Arnoldi Algorithm, build the Krylov space of orthonormal vectors
H = np.zeros((5, 4), dtype=complex)
q2bar = A @ q1
H[0, 0] = np.conjugate(q1).T @ q2bar
print(f"H_(0,0) = {H[0,0]}")
q2bar -= H[0, 0] * q1
dot_check1 = np.conjugate(q1).T @ q2bar
print(f"dot check1 = {dot_check1}")
H[1, 0] = np.sqrt(np.conjugate(q2bar).T @ q2bar)
q2 = q2bar / H[1, 0]
print(f"q2 = {q2}")
print(f"norm q2 = {np.linalg.norm(q2)}")

q3bar = A @ q2
H[0, 1] = np.conjugate(q1).T @ q3bar
H[1, 1] = np.conjugate(q2).T @ q3bar
q3bar -= H[0, 1] * q1
q3bar -= H[1, 1] * q2
dot_check2 = np.conjugate(q1).T @ q3bar
dot_check3 = np.conjugate(q2).T @ q3bar
print(f"dot check 2 = {dot_check2}")
print(f"dot check 3 = {dot_check3}")
H[2, 1] = np.sqrt(np.conjugate(q3bar).T @ q3bar)
q3 = q3bar / H[2, 1]
print(f"q3 = {q3}")
print(f"q3 norm = {np.linalg.norm(q3)}")

q4bar = A @ q3
H[0, 2] = np.conjugate(q1).T @ q4bar
H[1, 2] = np.conjugate(q2).T @ q4bar
H[2, 2] = np.conjugate(q3).T @ q4bar
q4bar -= H[0, 2] * q1
q4bar -= H[1, 2] * q2
q4bar -= H[2, 2] * q3
dot_check4 = np.conjugate(q1).T @ q4bar
dot_check5 = np.conjugate(q2).T @ q4bar
dot_check6 = np.conjugate(q3).T @ q4bar
print(f"dot check 4 = {dot_check4}")
print(f"dot check 5 = {dot_check5}")
print(f"dot check 6 = {dot_check6}")
H[3, 2] = np.sqrt(np.conjugate(q4bar).T @ q4bar)
q4 = q4bar / H[3, 2]
print(f"q4 = {q4}")
print(f"q4 norm = {np.linalg.norm(q4)}")

q5bar = A @ q4
H[0, 3] = np.conjugate(q1).T @ q5bar
H[1, 3] = np.conjugate(q2).T @ q5bar
H[2, 3] = np.conjugate(q3).T @ q5bar
H[3, 3] = np.conjugate(q4).T @ q5bar
q5bar -= H[0, 3] * q1
q5bar -= H[1, 3] * q2
q5bar -= H[2, 3] * q3
q5bar -= H[3, 3] * q4
print(f"q5bar = {q5bar}")
H[4, 3] = np.sqrt(np.conjugate(q5bar).T @ q5bar)

# perform the Gibbon's rotations
