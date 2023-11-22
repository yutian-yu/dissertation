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

while True:
    nu_1 = np.random.randn(4, )
    nu_1 = nu_1 / np.linalg.norm(nu_1)

    nu_2 = np.random.randn(4, )
    nu_2 = nu_2 / np.linalg.norm(nu_2)
    if np.dot(nu_1,nu_2) != 0 and np.dot(np.matmul(i,nu_1),nu_2) > 0:
        break

while True:
    nu_star = np.random.randn(4,)
    nu_star /= np.linalg.norm(nu_star)
    if np.dot(nu_star, nu_1) ** 2 + np.dot(nu_star, np.matmul(i, nu_1)) ** 2 != 0 and np.dot(nu_star, nu_2) ** 2 + np.dot(nu_star, np.matmul(i, nu_2)) ** 2 != 0:
        break

while True:
    u_star = np.random.randn(4,)
    u_star = u_star - np.dot(u_star, nu_star) * nu_star
    u_star = u_star - np.dot(u_star, np.matmul(i, nu_star)) * np.matmul(i, nu_star)
    u_star /= np.linalg.norm(u_star)
    if np.any(u_star):
        break
J_0_u_star = np.matmul(i, u_star)


def find_u_2(u_1, nu_1, nu_2):
    list_of_u_2 = []
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

    beta_1 = (- b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    alpha_1 = (- np.dot(np.matmul(i, nu_1), nu_2) - beta_1 * np.dot(u_1, temp_u_2)) / np.dot(np.matmul(i, u_1), temp_u_2)
    beta_2 = (- b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    alpha_2 = (- np.dot(np.matmul(i, nu_1), nu_2) - beta_2 * np.dot(u_1, temp_u_2)) / np.dot(np.matmul(i, u_1), temp_u_2)

    u_2 = alpha_1 * temp_u_2 + beta_1 * np.matmul(i, temp_u_2)
    list_of_u_2.append(u_2)
    u_2 = alpha_2 * temp_u_2 + beta_2 * np.matmul(i, temp_u_2)
    list_of_u_2.append(u_2)
    return list_of_u_2

u_11 = find_u_2(u_star, nu_star, nu_1)[0]
u_12 = find_u_2(u_star, nu_star, nu_1)[1]

u_21 = find_u_2(u_star, nu_star, nu_2)[0]
u_22 = find_u_2(u_star, nu_star, nu_2)[1]

def find_transition_matrix(u_1, J_0_u_1, u_2, J_0_u_2, nu_1, nu_2):
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

find_transition_matrix(u_11, np.matmul(i, u_11), u_21, np.matmul(i, u_21), nu_1, nu_2)
find_transition_matrix(u_12, np.matmul(i, u_12), u_21, np.matmul(i, u_21), nu_1, nu_2)
find_transition_matrix(u_11, np.matmul(i, u_11), u_22, np.matmul(i, u_22), nu_1, nu_2)
find_transition_matrix(u_12, np.matmul(i, u_12), u_22, np.matmul(i, u_22), nu_1, nu_2)