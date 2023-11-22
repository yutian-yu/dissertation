import numpy as np
import math as math
import random
from scipy.linalg import solve_triangular
from scipy.linalg import polar
from sklearn.preprocessing import normalize

# if the change of basis matrix in xi_0 is in SU(2), then what is the transition matrix from xi_1 to xi_2?

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

i2 = np.array([[0, -1],
               [1, 0]])

while True:
    su = np.random.randn(4,)
    su /= np.linalg.norm(su)
    if np.any(su):
        break
su1 = su[0]
su2 = su[1]
su3 = su[2]
su4 = su[3]

SU_2 = np.array([[su1, -su2, -su3, -su4],
                 [su2, su1, su4, -su3],
                 [su3, -su4, su1, su2],
                 [su4, su3, -su2, su1]])

def xi_to_xi(w, normal_vector_1, normal_vector_2):
    xi_to_xi = w - 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.matmul(i, normal_vector_1) - np.dot(w, np.matmul(i, normal_vector_2)) * np.matmul(i, normal_vector_2) + 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.dot(normal_vector_1,normal_vector_2) * np.matmul(i, normal_vector_2)
    return xi_to_xi

d = 6

# the only nu_1
while True:
    nu_1 = np.random.randn(d, )
    nu_1 = nu_1 / np.linalg.norm(nu_1)
    if np.any(nu_1):
        break

J_0_nu_1 = np.matmul(i, nu_1)

# first nu_2
while True:
    nu_2 = np.random.randn(d, )
    nu_2 = nu_2 / np.linalg.norm(nu_2)
    if np.any(nu_2) and np.dot(np.matmul(i,nu_1),nu_2) != 0 and np.dot(nu_1, nu_2) != 0:
        break

list_of_nu_2 = []
list_of_nu_2.append(nu_2)


list_of_w_2 = []
list_of_u_1 = []
list_of_u_2 = []

# the first w_2 to the first nu_2
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

list_of_w_2.append(w_2)

# the first u_1
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

list_of_u_1.append(u_1)

# starting the 2nd with the 2nd nu_2
while True:
    nu_2 = np.random.randn(d, )
    nu_2 = nu_2 / np.linalg.norm(nu_2)
    if np.any(nu_2) and np.dot(np.matmul(i,nu_1),nu_2) != 0 and np.dot(nu_1, nu_2) != 0 and np.dot(np.matmul(i, list_of_nu_2[0]), nu_2) > 0:
        break

list_of_nu_2.append(nu_2)



basis_mat = np.concatenate((u_1[:, np.newaxis], w_2[:, np.newaxis], J_0_u_1[:, np.newaxis], J_0_w_2[:, np.newaxis]), axis=1)
new_basis_mat = np.matmul(basis_mat, np.linalg.inv(SU_2))
print(new_basis_mat)
new_u_1 = new_basis_mat[0:6, 0]
new_w_2 = new_basis_mat[0:6, 1]
new_J_0_u_1 = new_basis_mat[0:6, 2]
new_J_0_w_2 = new_basis_mat[0:6, 3]

list_of_u_1.append(new_u_1)
list_of_w_2.append(new_w_2)

l = 0

for nu_2 in list_of_nu_2:
    J_0_nu_2 = np.matmul(i, nu_2)

    u_1 = list_of_u_1[l]
    J_0_u_1 = np.matmul(i, u_1)
    w_2 = list_of_w_2[l]
    J_0_w_2 = np.matmul(i, w_2)

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

    list_of_u_2.append(u_2)

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
    tr = np.trace(transition_matrix)
    u, p = polar(transition_matrix)

    print('transition matrix: \n{}'.format(transition_matrix))

    def row_col_swap (M):
        M[[1, 2], :] = M[[2, 1], :]
        M[:, [1, 2]] = M[:, [2, 1]]
        return M

    print('the trace is: {}'.format(tr))
    print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
    print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
    print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
    print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
    print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
    print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))
    print('polar decomp: u:\n{}'.format(u))
    print('p\n{}'.format(p))

    swapped_transition_matrix = row_col_swap(transition_matrix)

    print('swapped tran mat: \n{}'.format(swapped_transition_matrix))
    print('det of swapped: {}'.format(np.linalg.det(swapped_transition_matrix)))
    sliced_top_left = swapped_transition_matrix[0:2, 0:2]
    print('sliced top left: \n{}'.format(sliced_top_left))

    print('trace of sliced: {}'.format(np.trace(sliced_top_left)))
    print('det of sliced: {}'.format(np.linalg.det(sliced_top_left)))
    print('eigenvalues: {}'.format(np.linalg.eig(sliced_top_left)))
    print('DET OF (sliced - Id): {}'.format(np.linalg.det(sliced_top_left - np.identity(2))))
    print('ORTH? (want 0) (1)\n{}'.format(np.matmul(sliced_top_left, sliced_top_left.T)-np.identity(2)))
    print('\n (want 0) (2)\n{}'.format(np.matmul(sliced_top_left.T, sliced_top_left)-np.identity(2)))
    print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(sliced_top_left.T, np.linalg.inv(i2)), sliced_top_left), i2)-np.identity(2)))
    u, p = polar(sliced_top_left)
    print('polar decomp: u:\n{}'.format(u))
    print('p\n{}'.format(p))
    while True:
        v = np.random.randn(2,)
        if np.any(v):
            break
    print('det of [v, sliced*v]: {}'.format(np.linalg.det(np.concatenate((v[:, np.newaxis],np.matmul(sliced_top_left, v)[:, np.newaxis]),axis=1))))
    l += 1


M = np.concatenate((list_of_u_2[1][:, np.newaxis], list_of_w_2[1][:, np.newaxis], np.matmul(i, list_of_u_2[1])[:, np.newaxis], np.matmul(i, list_of_w_2[1])[:, np.newaxis]), axis=1)
Q, R = np.linalg.qr(M)

column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(list_of_u_2[0], list_of_nu_2[0], list_of_nu_2[1])), lower=False)
column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(list_of_w_2[0], list_of_nu_2[0], list_of_nu_2[1])), lower=False)
column_3 = solve_triangular(R, Q.T.dot(xi_to_xi(np.matmul(i, list_of_u_2[0]), list_of_nu_2[0], list_of_nu_2[1])), lower=False)
column_4 = solve_triangular(R, Q.T.dot(xi_to_xi(np.matmul(i, list_of_w_2[0]), list_of_nu_2[0], list_of_nu_2[1])), lower=False)

transition_matrix = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis], column_3[:, np.newaxis], column_4[:, np.newaxis]), axis=1)
tr = np.trace(transition_matrix)
u, p = polar(transition_matrix)

print('-----------------tran mat from 1 to 2------------------')
print('transition matrix: \n{}'.format(transition_matrix))
def row_col_swap (M):
        M[[1, 2], :] = M[[2, 1], :]
        M[:, [1, 2]] = M[:, [2, 1]]
        return M

print('the trace is: {}'.format(tr))
print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))
print('polar decomp: u:\n{}'.format(u))
print('p\n{}'.format(p))

swapped_transition_matrix = row_col_swap(transition_matrix)

print('swapped tran mat: \n{}'.format(swapped_transition_matrix))
print('det of swapped: {}'.format(np.linalg.det(swapped_transition_matrix)))
sliced_top_left = swapped_transition_matrix[0:2, 0:2]
print('sliced top left: \n{}'.format(sliced_top_left))

print('trace of sliced: {}'.format(np.trace(sliced_top_left)))
print('det of sliced: {}'.format(np.linalg.det(sliced_top_left)))
print('eigenvalues: {}'.format(np.linalg.eig(sliced_top_left)))
print('DET OF (sliced - Id): {}'.format(np.linalg.det(sliced_top_left - np.identity(2))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(sliced_top_left, sliced_top_left.T)-np.identity(2)))
print('\n (want 0) (2)\n{}'.format(np.matmul(sliced_top_left.T, sliced_top_left)-np.identity(2)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(sliced_top_left.T, np.linalg.inv(i2)), sliced_top_left), i2)-np.identity(2)))
u, p = polar(sliced_top_left)
print('polar decomp: u:\n{}'.format(u))
print('p\n{}'.format(p))
while True:
    v = np.random.randn(2,)
    if np.any(v):
        break
print('det of [v, sliced*v]: {}'.format(np.linalg.det(np.concatenate((v[:, np.newaxis],np.matmul(sliced_top_left, v)[:, np.newaxis]),axis=1))))

print('---------------change of basis mat in xi_star----------')
M = np.concatenate((list_of_u_1[1][:, np.newaxis], list_of_w_2[1][:, np.newaxis], np.matmul(i, list_of_u_1[1])[:, np.newaxis], np.matmul(i, list_of_w_2[1])[:, np.newaxis]), axis=1)
Q, R = np.linalg.qr(M)

column_1 = solve_triangular(R, Q.T.dot(list_of_u_1[0]), lower=False)
column_2 = solve_triangular(R, Q.T.dot(list_of_w_2[0]), lower=False)
column_3 = solve_triangular(R, Q.T.dot(np.matmul(i, list_of_u_1[0])), lower=False)
column_4 = solve_triangular(R, Q.T.dot(np.matmul(i, list_of_w_2[0])), lower=False)

transition_matrix = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis], column_3[:, np.newaxis], column_4[:, np.newaxis]), axis=1)
tr = np.trace(transition_matrix)
u, p = polar(transition_matrix)

print('transition matrix: \n{}'.format(transition_matrix))
def row_col_swap (M):
        M[[1, 2], :] = M[[2, 1], :]
        M[:, [1, 2]] = M[:, [2, 1]]
        return M

print('the trace is: {}'.format(tr))
print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))
print('polar decomp: u:\n{}'.format(u))
print('p\n{}'.format(p))

swapped_transition_matrix = row_col_swap(transition_matrix)

print('swapped tran mat: \n{}'.format(swapped_transition_matrix))
print('det of swapped: {}'.format(np.linalg.det(swapped_transition_matrix)))
sliced_top_left = swapped_transition_matrix[0:2, 0:2]
print('sliced top left: \n{}'.format(sliced_top_left))

print('trace of sliced: {}'.format(np.trace(sliced_top_left)))
print('det of sliced: {}'.format(np.linalg.det(sliced_top_left)))
print('eigenvalues: {}'.format(np.linalg.eig(sliced_top_left)))
print('DET OF (sliced - Id): {}'.format(np.linalg.det(sliced_top_left - np.identity(2))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(sliced_top_left, sliced_top_left.T)-np.identity(2)))
print('\n (want 0) (2)\n{}'.format(np.matmul(sliced_top_left.T, sliced_top_left)-np.identity(2)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(sliced_top_left.T, np.linalg.inv(i2)), sliced_top_left), i2)-np.identity(2)))
u, p = polar(sliced_top_left)
print('polar decomp: u:\n{}'.format(u))
print('p\n{}'.format(p))
while True:
    v = np.random.randn(2,)
    if np.any(v):
        break
print('det of [v, sliced*v]: {}'.format(np.linalg.det(np.concatenate((v[:, np.newaxis],np.matmul(sliced_top_left, v)[:, np.newaxis]),axis=1))))