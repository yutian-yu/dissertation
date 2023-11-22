import numpy as np
import math as math
import random
from scipy.linalg import solve_triangular
from sklearn.preprocessing import normalize



i = np.array([[0, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, -1],
              [1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0]])

i4 = np.array([[0, 0, -1, 0],
              [0, 0, 0, -1],
              [1, 0, 0, 0],
              [0, 1, 0, 0]])

def xi_to_xi(w, normal_vector_1, normal_vector_2):
    xi_to_xi = w - 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.matmul(i, normal_vector_1) - np.dot(w, np.matmul(i, normal_vector_2)) * np.matmul(i, normal_vector_2) + 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.dot(normal_vector_1,normal_vector_2) * np.matmul(i, normal_vector_2)
    return xi_to_xi

e1 = np.array([1, 0, 0, 0, 0, 0])
e2 = np.array([0, 1, 0, 0, 0, 0])
e3 = np.array([0, 0, 1, 0, 0, 0])
e4 = np.array([0, 0, 0, 1, 0, 0])
e5 = np.array([0, 0, 0, 0, 1, 0])
e6 = np.array([0, 0, 0, 0, 0, 1])

nu_1 = e1
J_0_nu_1 = e4

nu_2 = (1/2) * (e1 + e2 + e4 + e5)
J_0_nu_2 = (1/2) * (- e1 - e2 + e4 + e5)
'''
w_2 = e3
J_0_w_2 = np.matmul(i, w_2)

u_1 = e2
J_0_u_1 = np.matmul(i, u_1)

u_2 = (1/2) * (e1 - e2 + e4 - e5)
J_0_u_2 = np.matmul(i, u_2)'''

while True:
    part_nu_2 = nu_2 - np.dot(nu_2, nu_1) * nu_1
    orth_to_nu_1 = part_nu_2 - np.dot(part_nu_2, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
    orth_to_nu_1 /= np.linalg.norm(orth_to_nu_1)
    J_0_orth_to_nu_1 = np.matmul(i, orth_to_nu_1)

    w_2 = np.random.randn(6,)
    w_2 -= np.dot(w_2, nu_1) * nu_1
    w_2 -= np.dot(w_2, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
    w_2 -= np.dot(w_2, orth_to_nu_1) * orth_to_nu_1
    w_2 -= np.dot(w_2, np.matmul(i, orth_to_nu_1)) * np.matmul(i, orth_to_nu_1)
    w_2 /= np.linalg.norm(w_2)
    if np.any(w_2):
        break
J_0_w_2 = np.matmul(i, w_2)

print('w_2: {}'.format(w_2))
print(np.dot(w_2, nu_1))
print(np.dot(w_2, J_0_nu_1))
print(np.dot(w_2, nu_2))
print(np.dot(w_2, J_0_nu_2))

while True:
    u_1 = np.random.randn(6,)
    u_1 -= np.dot(u_1, nu_1) * nu_1
    u_1 -= np.dot(u_1, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
    u_1 -= np.dot(u_1, w_2) * w_2
    u_1 -= np.dot(u_1, np.matmul(i, w_2)) * np.matmul(i, w_2)
    u_1 /= np.linalg.norm(u_1)
    if np.any(u_1):
        break
J_0_u_1 = np.matmul(i, u_1)

print('u_1 : {}'.format(u_1))
print(np.dot(u_1, w_2))
print(np.dot(u_1, J_0_w_2))
print(np.dot(u_1, nu_1))
print(np.dot(u_1, J_0_nu_1))

while True:
    temp_u_2 = np.random.randn(6,)
    temp_u_2 -= np.dot(temp_u_2, nu_2) * nu_2
    temp_u_2 -= np.dot(temp_u_2, np.matmul(i, nu_2)) * np.matmul(i, nu_2)
    temp_u_2 -= np.dot(temp_u_2, w_2) * w_2
    temp_u_2 -= np.dot(temp_u_2, np.matmul(i, w_2)) * np.matmul(i, w_2)
    temp_u_2 /= np.linalg.norm(temp_u_2)
    if np.any(temp_u_2):
        break
J_0_temp_u_2 = np.matmul(i, temp_u_2)

C = np.array([[np.dot(u_1, temp_u_2), np.dot(u_1, np.matmul(i, temp_u_2))],
                [np.dot(np.matmul(i, u_1), temp_u_2), np.dot(u_1, temp_u_2)]])
a_b = np.linalg.solve(C, np.array([np.dot(nu_1, nu_2), - np.dot(np.matmul(i, nu_1), nu_2)]))

alpha = list(a_b)[0]
beta = list(a_b)[1]
print('1??? {}'.format(alpha**2 + beta**2))
u_2 = alpha*temp_u_2 + beta*J_0_temp_u_2
J_0_u_2= np.matmul(i, u_2)

print('u_2: {}'.format(u_2))
print(np.dot(u_2, nu_2))
print(np.dot(u_2, np.matmul(i, nu_2)))
print(np.dot(u_2, w_2))
print(np.dot(u_2, np.matmul(i, w_2)))

print('<u_1, u_2>: {}'.format(np.dot(u_1, u_2)))
print('<nu_1, nu_2>: {}'.format(np.dot(nu_1, nu_2)))
print('<1*u_1, u_2>: {}'.format(np.dot(np.matmul(i, u_1), u_2)))
print('<i*nu_1, nu_2>: {}'.format(np.dot(np.matmul(i, nu_1), nu_2)))

M = np.concatenate((u_2[:, np.newaxis], w_2[:, np.newaxis], J_0_u_2[:, np.newaxis], J_0_w_2[:, np.newaxis]), axis=1)
Q, R = np.linalg.qr(M)

column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(u_1, nu_1, nu_2)), lower=False)
column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(w_2, nu_1, nu_2)), lower=False)
column_3 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_u_1, nu_1, nu_2)), lower=False)
column_4 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_w_2, nu_1, nu_2)), lower=False)

transition_matrix = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis], column_3[:, np.newaxis], column_4[:, np.newaxis]), axis=1)
print('transition matrix: \n{}'.format(transition_matrix))
tr = np.trace(transition_matrix)
print('the trace is: {}'.format(tr))
print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))

print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))

