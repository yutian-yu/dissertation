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

'''
while True:
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
while True:
    nu_1 = np.random.randn(4, )
    nu_1 = nu_1 / np.linalg.norm(nu_1)
    if np.any(nu_1):
        break

nu_1st = nu_1

while True:
    u_1 = np.random.randn(4,)
    u_1 = u_1 - np.dot(u_1, nu_1) * nu_1
    u_1 = u_1 - np.dot(u_1, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
    u_1 /= np.linalg.norm(u_1)
    if np.any(u_1):
        break

u_1st = u_1

transition_matrix_1 = np.identity(2)

n = 3

for e in range(n):
    J_0_u_1 = np.matmul(i, u_1)
    while True:
        nu_2 = np.random.randn(4, )
        nu_2 = nu_2 / np.linalg.norm(nu_2)
        if np.dot(nu_1,nu_2) != 0 and np.dot(np.matmul(i,nu_1),nu_2) > 0:
            break

    '''
    nu_1 = np.array([1,0,1,0])
    nu_2 = np.array([0,1,1,0])

    nu_1 = nu_1 / np.linalg.norm(nu_1)
    nu_2 = nu_2 / np.linalg.norm(nu_2)
    '''

    while True:
        temp_u_2 = np.random.randn(4,)
        temp_u_2 -= np.dot(temp_u_2, nu_2) * nu_2
        temp_u_2 -= np.dot(temp_u_2, np.matmul(i, nu_2)) * np.matmul(i, nu_2)
        temp_u_2 /= np.linalg.norm(temp_u_2)
        if np.any(temp_u_2):
            break
    J_0_temp_u_2 = np.matmul(i, temp_u_2)

    a = np.dot(u_1, temp_u_2) ** 2 / (np.dot(np.matmul(i, u_1), temp_u_2) ** 2) + 1
    b = -2 * (- np.dot(np.matmul(i, nu_1), nu_2)) * np.dot(np.matmul(i, u_1), np.matmul(i, temp_u_2)) / (np.dot(np.matmul(i, u_1), temp_u_2) ** 2)
    c = np.dot(np.matmul(i, nu_1), nu_2) ** 2 / (np.dot(np.matmul(i, u_1), temp_u_2) ** 2) - 1

    # always have two distinct points (alpha1, beta1) and (alpha2, beta2)
    beta_1 = (- b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    alpha_1 = (- np.dot(np.matmul(i, nu_1), nu_2) - beta_1 * np.dot(u_1, temp_u_2)) / np.dot(np.matmul(i, u_1), temp_u_2)
    beta_2 = (- b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    alpha_2 = (- np.dot(np.matmul(i, nu_1), nu_2) - beta_2 * np.dot(u_1, temp_u_2)) / np.dot(np.matmul(i, u_1), temp_u_2)

    '''
    print('beta_1: {}'.format(beta_1))
    print('alpha_1: {}'.format(alpha_1))
    print('beta_2: {}'.format(beta_2))
    print('alpha_2: {}'.format(alpha_2))
    print('b^2 - 4ac: {}'.format(b * b - 4 * a * c))'''
    print('---------------------------------------------------\n----------------------new edge--------------------------')
    print('--------------using alpha_1, beta_1----------------')
    
    u_2 = alpha_1 * temp_u_2 + beta_1 * np.matmul(i, temp_u_2)
    J_0_u_2 = np.matmul(i, u_2)
    
    print('nu_1, nu_2: {}, {}'.format(nu_1, nu_2))
    print('<i*nu_1, nu_2>: {}'.format(np.dot(np.matmul(i,nu_1),nu_2)))
    print('<i*u_1, u_2>: {}'.format(np.dot(np.matmul(i,u_1),u_2)))
    print('u_1, J_0 u_1: {}, {}'.format(u_1, J_0_u_1))
    print('u_2, J_0 u_2: {}, {}'.format(u_2, J_0_u_2))
    
    M = np.concatenate((u_2[:, np.newaxis], J_0_u_2[:, np.newaxis]), axis=1)
    Q, R = np.linalg.qr(M)

    column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(u_1, nu_1, nu_2)), lower=False)
    column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_u_1, nu_1, nu_2)), lower=False)

    transition_matrix_2 = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis]), axis=1)
    print('transition matrix: \n{}'.format(transition_matrix_2))

    tr = np.trace(transition_matrix_2)
    print('the trace is: {}'.format(tr))

    while True:
        v = np.random.randn(2,)
        if np.any(v):
            break

    transition_matrix_v = np.matmul(transition_matrix_2,v)

    v_transition_matrix_v = np.concatenate((v[:, np.newaxis],transition_matrix_v[:, np.newaxis]),axis=1)

    det = np.linalg.det(v_transition_matrix_v)
    print('determinant of [v,Av] is {}'.format(det))
    
    u_1_print = u_1

    if det > 0 and tr > -2 and tr < 2:
        u_1 = u_2
        u_last = u_2
        transition_matrix_1_candidate = np.matmul(transition_matrix_2, transition_matrix_1)
        print('Positive elliptic!')
    elif det < 0 and tr > -2 and tr < 2:
        print('Negative elliptic!')
    else:
        print('Not elliptic!')

    print('---------------using alpha_2, beta_2------------------') 

    u_2 = alpha_2 * temp_u_2 + beta_2 * np.matmul(i, temp_u_2)
    J_0_u_2 = np.matmul(i, u_2)

    print('nu_1, nu_2: {}, {}'.format(nu_1, nu_2))
    print('<i*nu_1, nu_2>: {}'.format(np.dot(np.matmul(i,nu_1),nu_2)))
    print('<i*u_1, u_2>: {}'.format(np.dot(np.matmul(i, u_1_print),u_2)))
    print('u_1, J_0 u_1: {}, {}'.format(u_1_print, np.matmul(i, u_1_print)))
    print('u_2, J_0 u_2: {}, {}'.format(u_2, J_0_u_2))

    M = np.concatenate((u_2[:, np.newaxis], J_0_u_2[:, np.newaxis]), axis=1)
    Q, R = np.linalg.qr(M)

    column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(u_1_print, nu_1, nu_2)), lower=False)
    column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(np.matmul(i, u_1_print), nu_1, nu_2)), lower=False)

    transition_matrix_2 = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis]), axis=1)
    print('transition matrix: \n{}'.format(transition_matrix_2))

    tr = np.trace(transition_matrix_2)
    print('the trace is: {}'.format(tr))

    while True:
        v = np.random.randn(2,)
        if np.any(v):
            break
    
    transition_matrix_v = np.matmul(transition_matrix_2, v)
    v_transition_matrix_v = np.concatenate((v[:, np.newaxis],transition_matrix_v[:, np.newaxis]),axis=1)
    det = np.linalg.det(v_transition_matrix_v)
    print('determinant of [v,Av] is {}'.format(det))

    if det > 0 and tr > -2 and tr < 2:
        u_1 = u_2 
        u_last = u_2
        transition_matrix_1 = np.matmul(transition_matrix_2, transition_matrix_1)
        print('Positive elliptic!')
    elif det < 0 and tr > -2 and tr < 2:
        transition_matrix_1 = transition_matrix_1_candidate
        print('Negative elliptic!')
    else:
        transition_matrix_1 = transition_matrix_1_candidate
        print('Not elliptic!')
    
    nu_1 = nu_2


print('product of matrices: \n{}'.format(transition_matrix_1))

