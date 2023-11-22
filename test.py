import numpy as np
import math as math
import random
from scipy.linalg import solve_triangular
from sklearn.preprocessing import normalize


i = np.array([[0, 0, -1, 0],
              [0, 0, 0, -1],
              [1, 0, 0, 0],
              [0, 1, 0, 0]])

j = np.array([[0, -1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, -1, 0]])

k = np.array([[0, 0, 0, -1],
              [0, 0, 1, 0],
              [0, -1, 0, 0],
              [1, 0, 0, 0]])

def xi_to_xi(w, normal_vector_1, normal_vector_2):
    xi_to_xi = w - 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.matmul(i, normal_vector_1) - np.dot(w, np.matmul(i, normal_vector_2)) * np.matmul(i, normal_vector_2) + 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.dot(normal_vector_1,normal_vector_2) * np.matmul(i, normal_vector_2)
    return xi_to_xi

'''while True:
    try:
        nu_1 = input("1st normal vector nu_1 (4 numbers with space inbetween): \n")
        nu_2 = input("2nd nomral vector nu_2 (4 numbers with space inbetween): \n")

        nu_1 = np.array(list(map(int,nu_1.split())))
        nu_2 = np.array(list(map(int,nu_2.split())))

        nu_1 = nu_1 / np.linalg.norm(nu_1)
        nu_2 = nu_2 / np.linalg.norm(nu_2)

        print('Normalized nu_1 and nu_2: \n{},\n{}'.format(nu_1,nu_2))

        if np.dot(nu_1,nu_2) == 0 or not(np.any(np.matmul(np.matmul(i,nu_1),nu_2))) or np.matmul(np.matmul(i,nu_1),nu_2) < 0:
            raise ValueError
    
    except ValueError:
        print("Sorry, try again.")
        continue
    else:
        break'''
list_of_tran_mat = []
for l in range(2):
    while True:
        nu_1 = np.random.randn(4, )
        nu_1 = nu_1 / np.linalg.norm(nu_1)

        nu_2 = np.random.randn(4, )
        nu_2 = nu_2 / np.linalg.norm(nu_2)
        if np.dot(nu_1,nu_2) != 0 and np.dot(np.matmul(i,nu_1),nu_2) > 0:
            break

    '''nu_1 = np.array([1,0,1,0])
    nu_2 = np.array([0,1,1,0])

    nu_1 = nu_1 / np.linalg.norm(nu_1)
    nu_2 = nu_2 / np.linalg.norm(nu_2)'''
    '''print('nu_1, nu_2: {}, {}'.format(nu_1, nu_2))
    print('norm of nu_1: {}'.format(np.linalg.norm(nu_1)))
    print('norm of nu_2: {}'.format(np.linalg.norm(nu_2)))
    print('----------------------------------------')
    print('<nu_1, nu_2>: {}'.format(np.dot(nu_1, nu_2)))
    print('<i*nu_1, nu_2>: {}'.format(np.dot(np.matmul(i,nu_1),nu_2)))'''
    '''print('sum of the above two: {}'.format(np.dot(nu_1, nu_2) ** 2 + np.dot(np.matmul(i,nu_1),nu_2) ** 2))'''

    while True:
        u_1 = np.random.randn(4,)
        u_1 = u_1 - np.dot(u_1, nu_1) * nu_1
        u_1 = u_1 - np.dot(u_1, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
        u_1 /= np.linalg.norm(u_1)
        if np.any(u_1):
            break
    J_0_u_1 = np.matmul(i, u_1)

    while True:
        temp_u_2 = np.random.randn(4,)
        temp_u_2 -= np.dot(temp_u_2, nu_2) * nu_2
        temp_u_2 -= np.dot(temp_u_2, np.matmul(i, nu_2)) * np.matmul(i, nu_2)
        temp_u_2 /= np.linalg.norm(temp_u_2)
        if np.any(temp_u_2):
            break
    J_0_temp_u_2 = np.matmul(i, temp_u_2)

    C = np.array([[np.dot(u_1, temp_u_2), np.dot(u_1, np.matmul(i, temp_u_2))],
                [np.dot(np.matmul(i, u_1), temp_u_2), np.dot(u_1, temp_u_2)]])
    a_b = np.linalg.solve(C, np.array([np.dot(nu_1, nu_2), - np.dot(np.matmul(i, nu_1), nu_2)]))
            
    alpha = list(a_b)[0]
    beta = list(a_b)[1]

    u_2 = alpha * temp_u_2 + beta * np.matmul(i, temp_u_2)
    J_0_u_2 = np.matmul(i, u_2)

    '''print('<i*u_1, u_2>: {}'.format(np.dot(np.matmul(i,u_1),u_2)))
    print('---------------------------------------------')
    print('u_1, J_0 u_1: {}, {}'.format(u_1, J_0_u_1))
    print('<u_1, nu_1>: {}'.format(np.dot(u_1, nu_1)))
    print('<u_1, J_0 nu_1>: {}'.format(np.dot(u_1, np.matmul(i, nu_1))))
    print('the normal of u_1: {}'.format(np.linalg.norm(u_1)))
    print('-------------------------------------------------')
    print('u_2, J_0 u_2: {}, {}'.format(u_2, J_0_u_2))
    print('<u_2, nu_2>: {}'.format(np.dot(u_2, nu_2)))
    print('<u_2, J_0 nu_2>: {}'.format(np.dot(u_2, np.matmul(i, nu_2))))
    print('the normal of u_2: {}'.format(np.linalg.norm(u_2)))
    print('-----------------------------------------------')
    print('<u_1, u_2>: {}'.format(np.dot(u_1, u_2)))
    print('2*<u_1, u_2>: {}'.format(2 * np.dot(u_1, u_2)))
    print('<u_1, nu_2>: {}'.format(np.dot(u_1, nu_2)))
    print('<u_1, J_0 nu_2>: {}'.format(np.dot(u_1, np.matmul(i, nu_2))))
    print('<u_2, nu_1>: {}'.format(np.dot(u_2, nu_1)))
    print('<u_2, J_0 nu_1>: {}'.format(np.dot(u_2, np.matmul(i,nu_1))))
    print('<u_1, nu_2><J_0 nu_1, u_2>: {}'.format(np.dot(u_1, nu_2) * np.dot(np.matmul(i,nu_1), u_2)))
    print('<J_0 u_1, nu_2><nu_1, u_2>: {}'.format(np.dot(np.matmul(i, u_1), nu_2) * np.dot(nu_1, u_2)))
    print('sum of the above two: {}'.format(np.dot(u_1, nu_2) * np.dot(np.matmul(i,nu_1), u_2) + np.dot(np.matmul(i, u_1), nu_2) * np.dot(nu_1, u_2)))
    print('sum over <i*u_1, u_2>: {}'.format((np.dot(u_1, nu_2) * np.dot(np.matmul(i,nu_1), u_2) + np.dot(np.matmul(i, u_1), nu_2) * np.dot(nu_1, u_2))/np.dot(np.matmul(i, u_1), u_2)))
    print('trace: {}'.format(2 * np.dot(u_1, u_2) + (np.dot(u_1, nu_2) * np.dot(np.matmul(i,nu_1), u_2) + np.dot(np.matmul(i, u_1), nu_2) * np.dot(nu_1, u_2))/np.dot(np.matmul(i, u_1), u_2)))
    print('--------------------------------------------------')'''

    M = np.concatenate((u_2[:, np.newaxis], J_0_u_2[:, np.newaxis]), axis=1)
    Q, R = np.linalg.qr(M)

    column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(u_1, nu_1, nu_2)), lower=False)
    column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_u_1, nu_1, nu_2)), lower=False)

    transition_matrix = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis]), axis=1)
    print('transition matrix: \n{}'.format(transition_matrix))

    tr = np.trace(transition_matrix)
    print('the trace is: {}'.format(tr))

    v = np.random.randn(2,)
    transition_matrix_v = np.matmul(transition_matrix,v)

    v_transition_matrix_v = np.concatenate((v[:, np.newaxis],transition_matrix_v[:, np.newaxis]),axis=1)

    det = np.linalg.det(v_transition_matrix_v)
    print('determinant of [v,Av] is {}'.format(det))

    if det > 0 and tr > -2 and tr < 2:
        print('Positive elliptic!')
    elif det < 0 and tr > -2 and tr < 2:
        print('Negative elliptic!')

    '''print('-----------------same matrix? yes----------------------')
    xi_to_xi_of_u_1 = xi_to_xi(u_1, nu_1, nu_2)
    print('image of u_1 via xi_to_xi: {}'.format(xi_to_xi_of_u_1))
    print('above image projects to u_2: {}'.format(np.dot(xi_to_xi_of_u_1, u_2)))
    print('above image projects to J_0 u_2: {}'.format(np.dot(xi_to_xi_of_u_1, J_0_u_2)))
    xi_to_xi_of_u_2 = xi_to_xi(J_0_u_1, nu_1, nu_2)
    print('image of u_2 via xi_to_xi: {}'.format(xi_to_xi_of_u_2))
    print('above image projects to u_2: {}'.format(np.dot(xi_to_xi_of_u_2, u_2)))
    print('above image projects to J_0 u_2: {}'.format(np.dot(xi_to_xi_of_u_2, J_0_u_2)))
    print('transition matrix 2: \n{}'.format(np.array([[np.dot(xi_to_xi_of_u_1, u_2),np.dot(xi_to_xi_of_u_2, u_2)],[np.dot(xi_to_xi_of_u_1, J_0_u_2),np.dot(xi_to_xi_of_u_2, J_0_u_2)]])))'''

    list_of_tran_mat.append(transition_matrix)

print('1st tran mat: {}'.format(list_of_tran_mat[0]))
print('2nd tran mat: {}'.format(list_of_tran_mat[1]))
print(np.matmul(list_of_tran_mat[0], list_of_tran_mat[1]))
print(np.matmul(list_of_tran_mat[1], list_of_tran_mat[0]))

