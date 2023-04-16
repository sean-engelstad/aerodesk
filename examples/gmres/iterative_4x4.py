# GMRES iterative

import numpy as np

# develop an iterative procedure to match gmres_ex_hard.py
# the hardcoded GMRES for a small 4x4 system

np.random.seed(123456)

N = 4
A = np.random.rand(N, N)
b = np.random.rand(N, 1)
x0 = b  # + np.ones((N,1))
x = x0
r = b - A @ x
beta = np.linalg.norm(r)

# arnoldi's method matrices
steps = N
Q = np.zeros((N, steps + 1))
H = np.zeros((steps + 1, steps))
tol = 1e-10

xi = np.zeros((steps + 1))
xi[0] = 1.0
cosines = np.zeros((steps))
sines = np.zeros((steps))

q = r / beta
Q[:, 0] = q[:, 0]

# Arnoldi's iteration
for k in range(steps):
    qbar = A @ q
    for i in range(k + 1):
        H[i, k] = qbar.T @ Q[:, i]
    for i in range(k + 1):
        qbar[:, 0] -= H[i, k] * Q[:, i]
    H[k + 1, k] = np.linalg.norm(qbar)
    if abs(H[k + 1, k]) < 1e-12:
        q = 0 * qbar
    else:
        q = qbar / H[k + 1, k]
    Q[:, k + 1] = q[:, 0]

    # apply previous Gibben's rotations to H
    for i in range(k):
        temp = H[i, k]
        ci = cosines[i]
        si = sines[i]
        H[i, k] = ci * temp + si * H[i + 1, k]
        H[i + 1, k] = -si * temp + ci * H[i + 1, k]

    # compute kth Gibben's rotation
    cosines[k] = np.abs(H[k, k]) / np.sqrt(H[k, k] ** 2 + H[k + 1, k] ** 2)
    sines[k] = cosines[k] * H[k + 1, k] / H[k, k]

    # perform kth Gibben's rotation to H and xi
    temp = H[k, k]
    ck = cosines[k]
    sk = sines[k]
    H[k, k] = ck * temp + sk * H[k + 1, k]
    H[k + 1, k] = -sk * temp + ck * H[k + 1, k]

    temp = xi[k]
    xi[k] = ck * temp + sk * xi[k + 1]
    xi[k + 1] = -sk * temp + ck * xi[k + 1]

    print(f"xi_k+1 = {xi[k+1]}")
    if abs(beta * xi[k + 1]) < tol:
        break

print(f"Q = {Q}")
print(f"H = {H}")
print(f"cosines = {cosines}")
print(f"sines = {sines}")
print(f"xi = {xi}")

# solve the upper triangular system
# RHS = beta*(F*xi)_kx1
by = beta * xi[: k + 1]
by = np.reshape(by, newshape=(k + 1, 1))
R = H[: k + 1, : k + 1]
print(f"R = {R}")
print(f"by = {by}")

# solve the upper triangular system
y = np.zeros((k + 1, 1))
for i in range(k, -1, -1):
    print(f"i = {i}")
    nright = k - i
    numerator = by[i, 0]
    for iright in range(1, nright + 1):
        print(f"iright = {iright}")
        numerator -= R[i, i + iright] * y[i + iright, 0]
    y[i, 0] = numerator / R[i, i]

# check the upper triangular system was solved
upper_triangular_diff = R @ y - by
print(f"y = {y}")
print(f"upper triangular system check = {upper_triangular_diff}")

# compute solution xk = x0 + Qk * yk
x = x0 + Q[:, : k + 1] @ y
r = b - A @ x
print(f"x = {x}")
print(f"r = {r}")
