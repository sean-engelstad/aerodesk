import numpy as np

# GMRES playground
# here to figure out how the GMRES algorithm works w/o using a class
# manually make the rotation matrices, solve the least-squares problem, etc.
# debug my main GMRES class with this

np.random.seed(123456)

A = np.random.rand(4,4)
b = np.random.rand(4,1)
x0 = b #+ np.ones((4,1))
x = x0
r = b - A @ x
beta = np.linalg.norm(r)

print(f"A = {A}")
print(f"b = {b}")

# arnoldi's method matrices
Q = np.zeros((4,5))
H = np.zeros((5,4))

q1 = r / beta
Q[:,0] = q1[:,0]
print(f"q1 = {q1}")

# first Arnoldi's iteration
q2bar = A @ q1
H[0,0] = q2bar.T @ q1
q2bar -= H[0,0] * q1
H[1,0] = np.linalg.norm(q2bar)
q2 = q2bar / H[1,0]
Q[:,1] = q2[:,0]

# second Arnoldi's iteration
q3bar = A @ q2
H[0,1] = q3bar.T @ q1
H[1,1] = q3bar.T @ q2
q3bar -= H[0,1] * q1
q3bar -= H[1,1] * q2
H[2,1] = np.linalg.norm(q3bar)
q3 = q3bar / H[2,1]
Q[:,2] = q3[:,0]

# third Arnoldi's iteration
q4bar = A @ q3
H[0,2] = q4bar.T @ q1
H[1,2] = q4bar.T @ q2
H[2,2] = q4bar.T @ q3
q4bar -= H[0,2] * q1
q4bar -= H[1,2] * q2
q4bar -= H[2,2] * q3
H[3,2] = np.linalg.norm(q4bar)
q4 = q4bar / H[3,2]
Q[:,3] = q4[:,0]

# fourth Arnoldi's iteration
q5bar = A @ q4
H[0,3] = q5bar.T @ q1
H[1,3] = q5bar.T @ q2
H[2,3] = q5bar.T @ q3
H[3,3] = q5bar.T @ q4
q5bar -= H[0,3] * q1
q5bar -= H[1,3] * q2
q5bar -= H[2,3] * q3
q5bar -= H[3,3] * q4
H[4,3] = np.linalg.norm(q5bar)
if np.abs(H[4,3]) < 1e-12:
    q5 = 0 * q5bar
else:
    q5 = q5bar / H[4,3]
Q[:,4] = q5[:,0]

print(f"Q = {Q}")
print(f"H = {H}")

# check the identity A Q_k = Q_k+1 * H_k+1,k
arnoldi_diff = A @ Q[:,:4] - Q @ H
arnoldi_norm = np.linalg.norm(arnoldi_diff) 
print(f"norm of arnoldi diff = {arnoldi_norm}")

# check the Arnoldi identity #2 A Q_k = Q_k H_k + h_k+1,k * q_k+1 * xi_k^T
xi = np.zeros((4,1))
xi[0] = 1.0
arnoldi_diff2 = A @ Q[:,:4] - Q[:,:4] @ H[:4,:] - H[4,3] * q5 @ xi.T
arnoldi_norm2 = np.linalg.norm(arnoldi_diff2)
print(f"norm of arnoldi diff 2 = {arnoldi_norm2}")

# now goal is to minimize least-squares problem r_0 - A Q_k y_k = r_0 - Q_k+1 H_k+1,k for some unknown y_k
# note that r_0 = beta Q_k xi_1 implies norm(Q_k+1 (beta xi_1 - H_k+1,k y_k ) )
xi = np.zeros((5,1))
xi[0] = 1.0

# Gibbon's rotations
# rotation matrix F_1
F1 = np.eye(5)
if H[0,0] != 0:
    c1 = np.abs(H[0,0]) / np.sqrt(H[0,0]**2 + H[1,0]**2)
    s1 = c1 * H[1,0] / H[0,0]
else:
    c1 = 0.0; s1 = 1.0
F1[0:2,0:2] = np.array([[c1,s1],[-s1,c1]])

Htemp = F1 @ H
print(f"F1 = {F1}")
print(f"Htemp1 = {Htemp}")

# rotation matrix F_2
F2 = np.eye(5)
if Htemp[1,1] != 0:
    c2 = np.abs(Htemp[1,1]) / np.sqrt(Htemp[1,1]**2 + Htemp[2,1]**2)
    s2 = c2 * Htemp[2,1] / Htemp[1,1]
else:
    c2 = 0.0; s2 = 1.0
F2[1:3,1:3] = np.array([[c2,s2],[-s2,c2]])

Htemp = F2 @ Htemp
print(f"F2 = {F2}")
print(f"Htemp2 = {Htemp}")

# rotation matrix F3
F3 = np.eye(5)
if Htemp[2,2] != 0:
    c3 = np.abs(Htemp[2,2]) / np.sqrt(Htemp[2,2]**2 + Htemp[3,2]**2)
    s3 = c3 * Htemp[3,2] / Htemp[2,2]
else:
    c3 = 0.0; s3 = 1.0
F3[2:4,2:4] = np.array([[c3,s3],[-s3,c3]])

Htemp = F3 @ Htemp
print(f"F3 = {F3}")
print(f"Htemp3 = {Htemp}")

# rotation matrix F4
F4 = np.eye(5)
if Htemp[3,3] != 0:
    c4 = np.abs(Htemp[3,3]) / np.sqrt(Htemp[3,3]**2 + Htemp[4,3]**2)
    s4 = c4 * Htemp[4,3] / Htemp[3,3]
else:
    c4 = 0.0; s4 = 1.0
F4[3:5,3:5] = np.array([[c4,s4],[-s4,c4]])

Htemp = F4 @ Htemp
print(f"F4 = {F4}")
print(f"Htemp4 = {Htemp}")

# now we have completed the full Gibbons rotation matrix
Ftot = F4 @ F3 @ F2 @ F1

# now solve R_kxk y = beta * (F xi)_kx1
# with some resizing from R k+1xk+1 to kxk
by = beta * Ftot @ xi
by = by[0:4]
R = Htemp[0:4,0:4]
print(f"R = {R}")
print(f"beta*(F*xi)_kx1 = {by}")

# solve the upper triangular system of equations
y = np.zeros((4,1))
y[3] = by[3] / R[3,3]
y[2] = (by[2] - y[3] * R[2,3]) / R[2,2]
y[1] = (by[1] - y[2] * R[1,2] - y[3] * R[1,3]) / R[1,1]
y[0] = (by[0] - y[1] * R[0,1] - y[2] * R[0,2] - y[3] * R[0,3]) / R[0,0]

# check the upper triangular system is solved
upp_triang_check = R @ y - by
print(f"y = {y}")
print(f"R y - beta*F*xi norm = {np.linalg.norm(upp_triang_check)}")

# now compute x = x0 + Qk * yk
x = x0 + Q[:4,:4] @ y
rk = b - A @ x
print(f"x = {x}")
print(f"rk = {rk}")