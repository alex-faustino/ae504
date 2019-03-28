import math
import numpy as np
from scipy import linalg

# problem 3
N = 10
A = np.array([[.994, .026, 0, -32.2],
              [-.994, .376, 820, 0],
              [0, -.002, .332, 0],
              [0, 0, 1, 1]])
B = np.array([0, -32.7, -2.08, 0])

x = np.array([0., 100., 0., .4])
u = np.array([])

Q = np.diag([0, 1, 0, 0])
R = np.diag([0])

# problem 4
# N = 2
# A = np.array([[1., 0.],
#               [1., 1.]])
# B = np.array([1., 0.])

# x = np.array([0., 1.])
# u = np.array([])

# Q = np.diag([0, 1])
# R = np.diag([0])

P = Q
F = np.array([])

for i in range(N, 0, -1):
     tempF = -linalg.inv(R + B@P@B)[0]*B@P@A
     P = A.T@P@A + A.T@P@B@tempF + Q
     F = np.append(F, tempF, axis=0)

F = np.flip(np.reshape(F, (N, -1)), 0)

for j in range(np.shape(F)[0]):
     temp_u = F[j]@x
     x = np.dot(A, x) + np.dot(B, temp_u)
     u = np.append(u, np.array([temp_u]))

print (np.reshape(u, (N,-1)))