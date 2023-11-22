import numpy as np
import math as math
import random
from scipy.linalg import solve_triangular
from scipy.linalg import polar
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

i2 = np.array([[0, -1],
               [1, 0]])

def xi_to_xi(w, normal_vector_1, normal_vector_2):
    xi_to_xi = w - 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.matmul(i, normal_vector_1) - np.dot(w, np.matmul(i, normal_vector_2)) * np.matmul(i, normal_vector_2) + 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.dot(normal_vector_1,normal_vector_2) * np.matmul(i, normal_vector_2)
    return xi_to_xi

def row_col_swap (M):
            M[[1, 2], :] = M[[2, 1], :]
            M[:, [1, 2]] = M[:, [2, 1]]
            return M

while True:
    try:
        d = int(input('Please enter the number of dimensions (4 or 6): '))
        if d != 4 and d != 6:
            raise ValueError
    except ValueError:
        print('Sorry try again.')
        continue
    else:
        break

if d == 4:
    i = i4
    i4 = i2

while True:
    try: 
        man_or_ran = input('Would you like to manually input two normal vectors or have the program randomize them? (m or r): ')
        if man_or_ran != 'm' and man_or_ran != 'r':
            raise ValueError
    except ValueError:
        print('Sorry, try again.')
        continue
    else:
        break

if man_or_ran == 'm':
    while True:
        try:
            print('Please manually input 2 normal vectors that satisfy <J_0 nu_1, nu_2> > 0 ')
            nu_1 = input("1st normal vector nu_1 (4 or 6 numbers with space inbetween, e.g., 1 3 0 9): \n")
            nu_2 = input("2nd nomral vector nu_2 (4 or 6 numbers with space inbetween): \n")

            nu_1 = np.array(list(map(int,nu_1.split())))
            nu_2 = np.array(list(map(int,nu_2.split())))

            nu_1 = nu_1 / np.linalg.norm(nu_1)
            nu_2 = nu_2 / np.linalg.norm(nu_2)

            print('Normalized nu_1 and nu_2: \n{},\n{}'.format(nu_1,nu_2))

            if not(np.any(np.matmul(np.matmul(i,nu_1),nu_2))) or np.matmul(np.matmul(i,nu_1),nu_2) < 0 or len(nu_1) != len(nu_2):
                raise ValueError
        
        except ValueError:
            print("Sorry, try again.")
            continue
        else:
            break
else:
    while True:
        nu_1 = np.random.randn(d, )
        nu_1 = nu_1 / np.linalg.norm(nu_1)

        nu_2 = np.random.randn(d, )
        nu_2 = nu_2 / np.linalg.norm(nu_2)
        if np.dot(np.matmul(i,nu_1),nu_2) > 0 and np.dot(nu_1, nu_2) != 0:
            break

J_0_nu_1 = np.matmul(i, nu_1)
J_0_nu_2 = np.matmul(i, nu_2)

while True:
    try: 
        l = int(input('With these two normal vectors fixed, how many times would you like to generate a different u_1? '))
    except ValueError:
        print('Sorry, try again.')
        continue
    else:
        break

for k in range(l):
    if d == 6:
        while True:
            part_nu_2 = nu_2 - np.dot(nu_2, nu_1) * nu_1
            orth_to_nu_1 = part_nu_2 - np.dot(part_nu_2, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
            orth_to_nu_1 /= np.linalg.norm(orth_to_nu_1)
            J_0_orth_to_nu_1 = np.matmul(i, orth_to_nu_1)

            w_2 = np.random.randn(d,)
            w_2 -= np.dot(w_2, nu_1) * nu_1
            w_2 -= np.dot(w_2, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
            w_2 -= np.dot(w_2, orth_to_nu_1) * orth_to_nu_1
            w_2 -= np.dot(w_2, np.matmul(i, orth_to_nu_1)) * np.matmul(i, orth_to_nu_1)
            w_2 /= np.linalg.norm(w_2)
            if np.any(w_2):
                break
        J_0_w_2 = np.matmul(i, w_2)

        print('w_2: {}'.format(w_2))
        print(np.dot(w_2, nu_1))           # checking if w_2 is in the intersection of xi_1 and xi_2
        print(np.dot(w_2, J_0_nu_1))
        print(np.dot(w_2, nu_2))
        print(np.dot(w_2, J_0_nu_2))

    while True:
        u_1 = np.random.randn(d,)
        u_1 -= np.dot(u_1, nu_1) * nu_1
        u_1 -= np.dot(u_1, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
        if d == 6:
            u_1 -= np.dot(u_1, w_2) * w_2
            u_1 -= np.dot(u_1, np.matmul(i, w_2)) * np.matmul(i, w_2)
        u_1 /= np.linalg.norm(u_1)
        if np.any(u_1):
            break
    J_0_u_1 = np.matmul(i, u_1)

    print('u_1 : {}'.format(u_1))          
    if d == 6:                                  # checking if u_1 is orthononal to the intersection of xi_1 and xi_2 in dimension 6
        print(np.dot(u_1, w_2))
        print(np.dot(u_1, J_0_w_2))
    print(np.dot(u_1, nu_1))                    # checking if u_1 is orthogonal to Span{nu_1, J_0 nu_1}
    print(np.dot(u_1, J_0_nu_1))

    while True:
        temp_u_2 = np.random.randn(d,)
        temp_u_2 -= np.dot(temp_u_2, nu_2) * nu_2
        temp_u_2 -= np.dot(temp_u_2, np.matmul(i, nu_2)) * np.matmul(i, nu_2)
        if d == 6:
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
    print('norm of u_2 {}'.format(alpha**2 + beta**2))              # checking if u_2 has length 1
    u_2 = alpha*temp_u_2 + beta*J_0_temp_u_2
    J_0_u_2= np.matmul(i, u_2)

    print('u_2: {}'.format(u_2))                            # checking if u_2 is orthogonal to Span{nu_2, J_0 nu_2}
    print(np.dot(u_2, nu_2))
    print(np.dot(u_2, np.matmul(i, nu_2)))
    if d == 6:
        print(np.dot(u_2, w_2))                             # checking if u_2 is orthogonal to the intersection of xi_1 and xi_2 in dimension 6
        print(np.dot(u_2, np.matmul(i, w_2)))

    print('<u_1, u_2>: {}'.format(np.dot(u_1, u_2)))           # verifying u_1 and u_2 satisfy the desired relation
    print('<nu_1, nu_2>: {}'.format(np.dot(nu_1, nu_2)))
    print('<J_0 u_1, u_2>: {}'.format(np.dot(np.matmul(i, u_1), u_2)))
    print('<J_0 nu_1, nu_2>: {}'.format(np.dot(np.matmul(i, nu_1), nu_2)))

    if d == 6:
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
        print('the trace is: {}'.format(tr))
        print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
        print('determinant: {}'.format(np.linalg.det(transition_matrix)))
        print('determiant of (transition matrix - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(d-2))))
        print('orthogonal? (yes if 0) (left and right multiplications) (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(d-2)))
        print('\n (yes if 0) (2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(d-2)))
        print('symplectic? (yes if 0)\n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(d-2)))
        print('polar decomp: \nu:\n{}'.format(u))
        print('p:\n{}'.format(p))
        
        swapped_transition_matrix = row_col_swap(transition_matrix)
        print('columns and rows swapped transition matrix: \n{}'.format(swapped_transition_matrix))
        print('determinant: {}'.format(np.linalg.det(swapped_transition_matrix)))

        sliced_top_left = swapped_transition_matrix[0:2, 0:2]
        print('sliced top left 2x2: \n{}'.format(sliced_top_left))
        print('trace: {}'.format(np.trace(sliced_top_left)))
        print('determinant: {}'.format(np.linalg.det(sliced_top_left)))
        print('eigenvalues: {}'.format(np.linalg.eig(sliced_top_left)))
        print('determinant of (sliced - Id): {}'.format(np.linalg.det(sliced_top_left - np.identity(2))))
        print('orthogonal? (yes if 0) (left and right multiplications) (1)\n{}'.format(np.matmul(sliced_top_left, sliced_top_left.T)-np.identity(2)))
        print('\n (yes if 0) (2)\n{}'.format(np.matmul(sliced_top_left.T, sliced_top_left)-np.identity(2)))
        print('symplectic? (yes if 0)\n{}'.format(np.matmul(np.matmul(np.matmul(sliced_top_left.T, np.linalg.inv(i2)), sliced_top_left), i2)-np.identity(2)))
        u, p = polar(sliced_top_left)
        print('polar decomp: \nu:\n{}'.format(u))
        print('p:\n{}'.format(p))
        while True:
            v = np.random.randn(2,)
            if np.any(v):
                break
        sliced_det = np.linalg.det(sliced_top_left)
        sliced_tr = np.trace(sliced_top_left)
        print('determinant of [v, sliced*v]: {}'.format(sliced_det))
        if sliced_det > 0 and sliced_tr > -2 and sliced_tr < 2:
            print('Positive elliptic!')
        elif sliced_det < 0 and sliced_tr > -2 and sliced_tr < 2:
            print('Negative elliptic!')
        else:
            print('Not elliptic!')
    else: 
        M = np.concatenate((u_2[:, np.newaxis], J_0_u_2[:, np.newaxis]), axis=1)
        Q, R = np.linalg.qr(M)

        column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(u_1, nu_1, nu_2)), lower=False)
        column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(np.matmul(i, u_1), nu_1, nu_2)), lower=False)

        transition_matrix_2 = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis]), axis=1)
        print('transition matrix: \n{}'.format(transition_matrix_2))
        print('eigenvalues: \n{}'.format(np.linalg.eig(transition_matrix_2)))
        tr = np.trace(transition_matrix_2)
        print('the trace is: {}'.format(tr))
        u, p = polar(transition_matrix_2)
        print('eigenvalues of u: {}'.format(np.linalg.eig(u)))
        while True:
            v = np.random.randn(2,)
            if np.any(v):
                break
        
        transition_matrix_v = np.matmul(transition_matrix_2, v)
        v_transition_matrix_v = np.concatenate((v[:, np.newaxis],transition_matrix_v[:, np.newaxis]),axis=1)
        det = np.linalg.det(v_transition_matrix_v)
        print('determinant of [v,Av] is {}'.format(det))

        if det > 0 and tr > -2 and tr < 2:
            print('Positive elliptic!')
        elif det < 0 and tr > -2 and tr < 2:
            print('Negative elliptic!')
        else:
            print('Not elliptic!')