import numpy as np, unittest

# generate a
N = 5
A = np.random.rand(N, N)
b = np.random.rand(N, 1)

x = 0.5 * b
r0 = b - A @ x
r = r0


def norm(x):
    return np.linalg.norm(x)


# choose a number of steps for the GMRES algorithm
steps = 4

# initial krylov unit vectors
q = r / norm(r)
xi = np.zeros((steps, 1))
beta = norm(r)

Q = np.zeros((N, steps + 1))
H = np.zeros((steps + 1, steps))

Q[:, 0] = q[:, 0]
print(f"Q0 = {Q}")

for k in range(steps):
    # arnoldi's method
    q = Q[:, k]
    qtemp = A @ q
    # compute kth column of H_{k+1,k}
    for i in range(k + 1):  # 0,1,2,...,k
        H[i, k] = qtemp.T @ Q[:, i]
    for i in range(k + 1):
        qtemp[:] -= H[i, k] * Q[:, i]
    H[k + 1, k] = norm(qtemp)
    Q[:, k + 1] = qtemp / H[k + 1, k]

# check arnoldi method
LHS = A @ Q[:, :steps]
RHS = Q[: steps + 1] @ H[: steps + 1, :steps]
arnoldi_diff = LHS - RHS
arnoldi_norm = np.linalg.norm(arnoldi_diff)
print(f"arnoldi norm = {arnoldi_norm}")
