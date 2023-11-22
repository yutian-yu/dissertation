import numpy as np
import math as math
import cmath
import random
from scipy.linalg import solve_triangular
from scipy.linalg import polar
from sklearn.preprocessing import normalize

# different basis of nu_star to nu_1 and nu_2. The compare the change of basis matrix in xi_star with the transition mat from nu_1 to nu_2

i = np.array([[0, -1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 0, 0, -1, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, -1],
              [0, 0, 0, 0, 1, 0]])

i4 = np.array([[0, -1, 0, 0],
               [1, 0, 0, 0],
               [0, 0, 0, -1],
               [0, 0, 1, 0]])

i2 = np.array([[0, -1],
               [1, 0]])

def xi_to_xi(w, normal_vector_1, normal_vector_2):
    xi_to_xi = w - 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.matmul(i, normal_vector_1) - np.dot(w, np.matmul(i, normal_vector_2)) * np.matmul(i, normal_vector_2) + 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.dot(normal_vector_1,normal_vector_2) * np.matmul(i, normal_vector_2)
    return xi_to_xi


def get_SU2():
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
    return SU_2

d = 6

while True:
    nu_1 = np.random.randn(d, )
    nu_1 = nu_1 / np.linalg.norm(nu_1)

    nu_2 = np.random.randn(d, )
    nu_2 = nu_2 / np.linalg.norm(nu_2)

    nu_3 = np.random.randn(d,)
    nu_3 /= np.linalg.norm(nu_3)
    if np.any(nu_1) and np.any(nu_2) and np.any(nu_3) and np.dot(np.matmul(i,nu_1),nu_2) > 0 and np.dot(np.matmul(i,nu_2),nu_3) > 0 and np.dot(nu_1, nu_2) != 0 and np.dot(nu_2, nu_3) != 0:
        break

list_of_nu = []
list_of_nu.append(nu_1)
list_of_nu.append(nu_2)
list_of_nu.append(nu_3)

list_of_w_2 = []
list_of_u_1 = []
list_of_u_2 = []
list_of_tran_mat = []


for l in range(2):
    nu_1 = list_of_nu[l]
    nu_2 = list_of_nu[l+1]
    J_0_nu_1 = np.matmul(i, nu_1)
    J_0_nu_2 = np.matmul(i, nu_2)

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

    '''print('w_2: {}'.format(w_2))
    print(np.dot(w_2, nu_1))
    print(np.dot(w_2, J_0_nu_1))
    print(np.dot(w_2, nu_2))
    print(np.dot(w_2, J_0_nu_2))'''

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

    '''print('u_1 : {}'.format(u_1))
    print(np.dot(u_1, w_2))
    print(np.dot(u_1, J_0_w_2))
    print(np.dot(u_1, nu_1))
    print(np.dot(u_1, J_0_nu_1))'''

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

    '''print('u_2: {}'.format(u_2))
    print(np.dot(u_2, nu_2))
    print(np.dot(u_2, np.matmul(i, nu_2)))
    print(np.dot(u_2, w_2))
    print(np.dot(u_2, np.matmul(i, w_2)))'''

    '''print('<u_1, u_2>: {}'.format(np.dot(u_1, u_2)))
    print('<nu_1, nu_2>: {}'.format(np.dot(nu_1, nu_2)))
    print('<1*u_1, u_2>: {}'.format(np.dot(np.matmul(i, u_1), u_2)))
    print('<i*nu_1, nu_2>: {}'.format(np.dot(np.matmul(i, nu_1), nu_2)))'''

    M = np.concatenate((u_2[:, np.newaxis], J_0_u_2[:, np.newaxis], w_2[:, np.newaxis], J_0_w_2[:, np.newaxis]), axis=1)
    Q, R = np.linalg.qr(M)

    column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(u_1, nu_1, nu_2)), lower=False)
    column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_u_1, nu_1, nu_2)), lower=False)
    column_3 = solve_triangular(R, Q.T.dot(xi_to_xi(w_2, nu_1, nu_2)), lower=False)
    column_4 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_w_2, nu_1, nu_2)), lower=False)

    transition_matrix = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis], column_3[:, np.newaxis], column_4[:, np.newaxis]), axis=1)

    # collecting 2 transition matrices
    copy_of_transition_matrix = transition_matrix.copy()
    list_of_tran_mat.append(copy_of_transition_matrix)

    tr = np.trace(transition_matrix)
    u, p = polar(transition_matrix)
    
    print('\nSTART OF PRINTING FOR A SINGLE TRANSITION MATRIX\n')
    print('transition matrix: \n{}'.format(transition_matrix))
    print('the trace is: {}'.format(tr))
    print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
    print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
    print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
    print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
    print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
    print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))
    print('polar decomp: u:\n{}'.format(u))
    print('eigenvalues of u: {}'.format(np.linalg.eig(u)))
    complex = np.array([[u[0, 0] + 1j*u[1, 0], 
                u[0, 2] + 1j*u[1, 2]],
                [u[2, 0] + 1j*u[3, 0], 
                u[2, 2] + 1j*u[3, 2]]])
    print('u in complex form: \n{}'.format(complex))
    print('eigenvalues of u in complex form: {}'.format(np.linalg.eig(complex)))
    print('complex det of u: {}'.format(complex[0, 0]*complex[1, 1] - complex[1, 0]*complex[0, 1]))
    print('p\n{}'.format(p))
    sliced_top_left_of_p = p[0:2, 0:2]
    print('eigenvalues of TOP LEFT OF P: \n{}'.format(np.linalg.eig(sliced_top_left_of_p)))
    print('ORTH? (want 0) (1)\n{}'.format(np.matmul(sliced_top_left_of_p, sliced_top_left_of_p.T)-np.identity(2)))
    print('\n (want 0) (2)\n{}'.format(np.matmul(sliced_top_left_of_p.T, sliced_top_left_of_p)-np.identity(2)))
    print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(sliced_top_left_of_p.T, np.linalg.inv(i2)), sliced_top_left_of_p), i2)-np.identity(2)))
    while True:
        v = np.random.randn(4,)
        if np.any(v):
            break
    print('is p positive matrix? {}'.format(np.matmul(v, np.matmul(p, v))))

    sliced_top_left = transition_matrix[0:2, 0:2]

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
    print('eigvals of u: {}'.format(np.linalg.eig(u)))
    print('p\n{}'.format(p))
    while True:
        v = np.random.randn(2,)
        if np.any(v):
            break
    print('det of [v, sliced*v]: {}'.format(np.linalg.det(np.concatenate((v[:, np.newaxis],np.matmul(sliced_top_left, v)[:, np.newaxis]),axis=1))))
    print('\nEND OF PRINTING FOR A SINGLE TRANSITION MATRIX\n')

# below matrix is from xi_1 to xi_2

transition_matrix_1to2 = list_of_tran_mat[0]
transition_matrix_2to3 = list_of_tran_mat[1]
'''print('---check the 1nd tran mat ----')
print('SAME? \n{}'.format(transition_matrix_1to2))'''

print('\n---------------change of basis mat in xi_2 ----------')
M = np.concatenate((list_of_u_1[1][:, np.newaxis], np.matmul(i, list_of_u_1[1])[:, np.newaxis], list_of_w_2[1][:, np.newaxis], np.matmul(i, list_of_w_2[1])[:, np.newaxis]), axis=1)
Q, R = np.linalg.qr(M)

column_1 = solve_triangular(R, Q.T.dot(list_of_u_2[0]), lower=False)
column_2 = solve_triangular(R, Q.T.dot(np.matmul(i, list_of_u_2[0])), lower=False)
column_3 = solve_triangular(R, Q.T.dot(list_of_w_2[0]), lower=False)
column_4 = solve_triangular(R, Q.T.dot(np.matmul(i, list_of_w_2[0])), lower=False)

transition_matrix = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis], column_3[:, np.newaxis], column_4[:, np.newaxis]), axis=1)

print('---check the unitary mat---')
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))

tr = np.trace(transition_matrix)
u, p = polar(transition_matrix)

print('transition matrix: \n{}'.format(transition_matrix))
print('the trace is: {}'.format(tr))
print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))
print('polar decomp: u:\n{}'.format(u))
print('p\n{}'.format(p))

complex = np.array([[transition_matrix[0, 0] + 1j*transition_matrix[1, 0], 
                            transition_matrix[0, 2] + 1j*transition_matrix[1, 2]],
                            [transition_matrix[2, 0] + 1j*transition_matrix[3, 0], 
                            transition_matrix[2, 2] + 1j*transition_matrix[3, 2]]])
print('in complex form: \n{}'.format(complex))
print('eigenvalues of complex form: {}'.format(np.linalg.eig(complex)))
print('complex det: {}'.format(complex[0, 0]*complex[1, 1] - complex[1, 0]*complex[0, 1]))
print('---------- END OF BASIS UNITARY MATRIX --------')

'''sliced_top_left = transition_matrix[0:2, 0:2]

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
print('det of [v, sliced*v]: {}'.format(np.linalg.det(np.concatenate((v[:, np.newaxis],np.matmul(sliced_top_left, v)[:, np.newaxis]),axis=1))))'''
print('---- list of w_2 ----')
print('1???????????: {}'.format(np.dot(list_of_w_2[0], list_of_w_2[1])**2 + np.dot(list_of_w_2[0], np.matmul(i, list_of_w_2[1]))**2))

# want to investigate the product matrix of the unitary matrix in xi_2 and the 1nd transition matrix from xi_1 to xi_2, so just updating the 'transition matrix' to be the product matrix

'''transition_matrix = np.matmul(transition_matrix, transition_matrix_1to2)
print('---------- below info is for the product matrix of the unitary matrix and the 2nd transition matrix -------')

tr = np.trace(transition_matrix)

print('transition matrix: \n{}'.format(transition_matrix))

print('the trace is: {}'.format(tr))
print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))

u, p = polar(transition_matrix)
print('polar decomp: u:\n{}'.format(u))
print('eigenvalues of u: {}'.format(np.linalg.eig(u)))
print('p\n{}'.format(p))
complex = np.array([[u[0, 0] + 1j*u[1, 0], 
                    u[0, 2] + 1j*u[1, 2]],
                    [u[2, 0] + 1j*u[3, 0], 
                    u[2, 2] + 1j*u[3, 2]]])
print('u in complex form: \n{}'.format(complex))
print('eigenvalues of u in complex form: {}'.format(np.linalg.eig(complex)))
print('complex det of u: {}'.format(complex[0, 0]*complex[1, 1] - complex[1, 0]*complex[0, 1]))'''

'''sliced_top_left = swapped_transition_matrix[0:2, 0:2]

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
print('---- list of w_2 ----')
print('1???????????: {}'.format(np.dot(list_of_w_2[0], list_of_w_2[1])**2 + np.dot(list_of_w_2[0], np.matmul(i, list_of_w_2[1]))**2))'''

# want to investigate the product matrix of a random SU(2) matrix in xi_2 and the 1nd transition matrix from xi_1 to xi_2, so just updating the 'transition matrix' to be the product matrix

'''print('---------- below info is for the product matrix of a random SU(2) matrix and the 2nd transition matrix -------')

tr = np.trace(transition_matrix)

print('transition matrix: \n{}'.format(transition_matrix))

print('the trace is: {}'.format(tr))
print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
print('DET OF transition matrix: {}'.format(np.linalg.det(transition_matrix)))
print('DET OF (tran_mat - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
print('ORTH? (want 0) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
print('\n (want 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
print('SYMP? (want 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))

u, p = polar(transition_matrix)
print('polar decomp: u:\n{}'.format(u))
print('eigenvalues of u: {}'.format(np.linalg.eig(u)))
print('p\n{}'.format(p))
complex = np.array([[u[0, 0] + 1j*u[1, 0], 
                    u[0, 2] + 1j*u[1, 2]],
                    [u[2, 0] + 1j*u[3, 0], 
                    u[2, 2] + 1j*u[3, 2]]])
print('u in complex form: \n{}'.format(complex))
print('eigenvalues of u in complex form: {}'.format(np.linalg.eig(complex)))
print('complex det of u: {}'.format(complex[0, 0]*complex[1, 1] - complex[1, 0]*complex[0, 1]))'''

'''# Applying rho hat map from Gutt
# single tran mat into rho hat
print('\n rho hat from Gutt')
print('\n single tran mat')
top = 1/2 * (transition_matrix_1to2 - np.matmul(np.matmul(i4, transition_matrix_1to2), i4))
print('1/2(tran_mat-J*tran_mat*J): \n{}'.format(top))
complex = np.array([[top[0, 0] + 1j*top[1, 0], 
                    top[0, 2] + 1j*top[1, 2]],
                    [top[2, 0] + 1j*top[3, 0], 
                    top[2, 2] + 1j*top[3, 2]]])
comp_det = np.linalg.det(complex)
print('complex det of top: {}'.format(comp_det))
print('normalized to the unit circle: {}'.format(comp_det / np.sqrt(comp_det.real**2 + comp_det.imag**2)))

# 1st product of SU(2) and block diag tran mat
print('\n 1st prod')
SU2 = get_SU2()
print('SU2 info')
complex = np.array([[SU2[0, 0] + 1j*SU2[1, 0], 
                    SU2[0, 2] + 1j*SU2[1, 2]],
                    [SU2[2, 0] + 1j*SU2[3, 0], 
                    SU2[2, 2] + 1j*SU2[3, 2]]])
print('complex SU2: \n{}'.format(complex))
print('eigenvalues of real SU2: {}'.format(np.linalg.eig(SU2)))
print('eigenvalues of complex SU2: {}'.format(np.linalg.eig(complex)))
transition_matrix = np.matmul(SU2, transition_matrix_1to2)
print('DET OF (1st prod - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
print('eigenvalues of the 1st prod: {}'.format(np.linalg.eig(transition_matrix)))
print('eigenvalues of 1st tran mat: {}'.format(np.linalg.eig(transition_matrix_1to2)))
top = 1/2 * (transition_matrix - np.matmul(np.matmul(i4, transition_matrix), i4))
print('1/2(prod-J*prod*J): \n{}'.format(top))
complex = np.array([[top[0, 0] + 1j*top[1, 0], 
                    top[0, 2] + 1j*top[1, 2]],
                    [top[2, 0] + 1j*top[3, 0], 
                    top[2, 2] + 1j*top[3, 2]]])
comp_det = np.linalg.det(complex)
print('complex det of top: {}'.format(comp_det))
z_1 = comp_det / np.sqrt(comp_det.real**2 + comp_det.imag**2)
print('normalized to the unit circle z_1: {}'.format(z_1))
u, p = polar(transition_matrix)
print('polar decom of 1st prod: u: \n{}'.format(u))
complex = np.array([[u[0, 0] + 1j*u[1, 0], 
                    u[0, 2] + 1j*u[1, 2]],
                    [u[2, 0] + 1j*u[3, 0], 
                    u[2, 2] + 1j*u[3, 2]]])
print('complex det of u: {}'.format(complex[0, 0]*complex[1, 1] - complex[1, 0]*complex[0, 1]))

# 2nd product of SU(2) and block diag tran mat
print('\n 2nd prod')
SU2 = get_SU2()
print('SU2 info')
complex = np.array([[SU2[0, 0] + 1j*SU2[1, 0], 
                    SU2[0, 2] + 1j*SU2[1, 2]],
                    [SU2[2, 0] + 1j*SU2[3, 0], 
                    SU2[2, 2] + 1j*SU2[3, 2]]])
print('complex SU2: \n{}'.format(complex))
print('eigenvalues of real SU2: {}'.format(np.linalg.eig(SU2)))
print('eigenvalues of complex SU2: {}'.format(np.linalg.eig(complex)))
prod_2 = np.matmul(SU2, transition_matrix_2to3)
print('DET OF (prod_2 - Id): {}'.format(np.linalg.det(prod_2 - np.identity(4))))
print('eigenvalues of the 2nd prod: {}'.format(np.linalg.eig(prod_2)))
print('eigenvalues of 2nd tran mat: {}'.format(np.linalg.eig(transition_matrix_2to3)))
top = 1/2 * (prod_2 - np.matmul(np.matmul(i4, prod_2), i4))
print('1/2(prod-J*prod*J): \n{}'.format(top))
complex = np.array([[top[0, 0] + 1j*top[1, 0], 
                    top[0, 2] + 1j*top[1, 2]],
                    [top[2, 0] + 1j*top[3, 0], 
                    top[2, 2] + 1j*top[3, 2]]])
comp_det = np.linalg.det(complex)
print('complex det of top: {}'.format(comp_det))
z_2 = comp_det / np.sqrt(comp_det.real**2 + comp_det.imag**2)
print('normalized to the unit circle z_2: {}'.format(z_2))
u, p = polar(prod_2)
print('polar decom of prod_2: u: \n{}'.format(u))
complex = np.array([[u[0, 0] + 1j*u[1, 0], 
                    u[0, 2] + 1j*u[1, 2]],
                    [u[2, 0] + 1j*u[3, 0], 
                    u[2, 2] + 1j*u[3, 2]]])
print('complex det of u: {}'.format(complex[0, 0]*complex[1, 1] - complex[1, 0]*complex[0, 1]))

# prod of the above two products
print('\n prod of two products')
A_psi_A_psi = np.matmul(prod_2, transition_matrix)
print('DET OF (A_psi_A_psi - Id): {}'.format(np.linalg.det(A_psi_A_psi - np.identity(4))))
top = 1/2 * (A_psi_A_psi - np.matmul(np.matmul(i4, A_psi_A_psi), i4))
print('1/2(2prods-J*2prods*J): \n{}'.format(top))
complex = np.array([[top[0, 0] + 1j*top[1, 0], 
                    top[0, 2] + 1j*top[1, 2]],
                    [top[2, 0] + 1j*top[3, 0], 
                    top[2, 2] + 1j*top[3, 2]]])
comp_det = np.linalg.det(complex)
print('complex det of top: {}'.format(comp_det))
z_3 = comp_det / np.sqrt(comp_det.real**2 + comp_det.imag**2)
print('normalized to the unit circle z_3: {}'.format(z_3))
print('\nIs the product of z_1 and z_2 equal to z_3? {}'.format(z_1 * z_2))
u, p = polar(A_psi_A_psi)
print('polar decom of A_psi_A_psi: u: \n{}'.format(u))
complex = np.array([[u[0, 0] + 1j*u[1, 0], 
                    u[0, 2] + 1j*u[1, 2]],
                    [u[2, 0] + 1j*u[3, 0], 
                    u[2, 2] + 1j*u[3, 2]]])
print('complex det of u: {}'.format(complex[0, 0]*complex[1, 1] - complex[1, 0]*complex[0, 1]))'''