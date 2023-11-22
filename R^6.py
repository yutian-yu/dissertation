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

'''j = np.array([[0, -1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, -1, 0]])'''

k = np.array([[0, 0, 0, -1],
              [0, 0, 1, 0],
              [0, -1, 0, 0],
              [1, 0, 0, 0]])

def xi_to_xi(w, normal_vector_1, normal_vector_2):
    xi_to_xi = w - 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.matmul(i, normal_vector_1) - np.dot(w, np.matmul(i, normal_vector_2)) * np.matmul(i, normal_vector_2) + 1 / np.dot(np.matmul(i,normal_vector_1),normal_vector_2) * np.dot(w, normal_vector_2) * np.dot(normal_vector_1,normal_vector_2) * np.matmul(i, normal_vector_2)
    return xi_to_xi

def mm(v_1,v_2):
    mm = np.matmul(v_1,v_2)
    return mm

def mm3(v_1, v_2, v_3):
    mm3 = np.matmul(np.matmul(v_1, v_2), v_3)
    return mm3

def imm(v_1,v_2):
    imm = np.dot(np.matmul(i,v_1),v_2)
    return imm

def dt(v_1,v_2):
    dt = np.dot(v_1,v_2)
    return dt

d = 6

while True:
    nu_1 = np.random.randn(d, )
    nu_1 = nu_1 / np.linalg.norm(nu_1)

    nu_2 = np.random.randn(d, )
    nu_2 = nu_2 / np.linalg.norm(nu_2)
    if np.dot(np.matmul(i,nu_1),nu_2) > 0 and np.dot(nu_1, nu_2) != 0:
        break

def get_alpha_beta(u_1, temp_u_2):
    a = np.dot(u_1, temp_u_2) ** 2 / (np.dot(np.matmul(i, u_1), temp_u_2) ** 2) + 1
    b = -2 * (- np.dot(np.matmul(i, nu_1), nu_2)) * np.dot(np.matmul(i, u_1), np.matmul(i, temp_u_2)) / (np.dot(np.matmul(i, u_1), temp_u_2) ** 2)
    c = np.dot(np.matmul(i, nu_1), nu_2) ** 2 / (np.dot(np.matmul(i, u_1), temp_u_2) ** 2) - 1

    b_1 = (- b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    a_1 = (- np.dot(np.matmul(i, nu_1), nu_2) - b_1 * np.dot(u_1, temp_u_2)) / np.dot(np.matmul(i, u_1), temp_u_2)
    b_2 = (- b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    a_2 = (- np.dot(np.matmul(i, nu_1), nu_2) - b_2 * np.dot(u_1, temp_u_2)) / np.dot(np.matmul(i, u_1), temp_u_2)
    return [a_1, b_1, a_2, b_2]

def get_alpha_beta_1(u_1, temp_u_2):
    a = np.dot(u_1, np.matmul(i, temp_u_2))**2 / np.dot(u_1, temp_u_2)**2 + 1
    b = - 2 * np.dot(nu_1,nu_2) * np.dot(u_1, np.matmul(i, temp_u_2)) / np.dot(u_1, temp_u_2)**2
    c = np.dot(nu_1,nu_2)**2 / np.dot(u_1,temp_u_2)**2 - 1

    b_1 = (- b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    a_1 = (np.dot(nu_1,nu_2)-b_1*np.dot(u_1,np.matmul(i,temp_u_2)))/np.dot(u_1, temp_u_2)
    b_2 = (- b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
    a_2 = (np.dot(nu_1,nu_2)-b_2*np.dot(u_1,np.matmul(i,temp_u_2)))/np.dot(u_1, temp_u_2)
    return [a_1, b_1, a_2, b_2]

def get_alpha_beta_2(u_1, temp_u_2):
    C = np.array([[np.dot(u_1, temp_u_2), np.dot(u_1, np.matmul(i, temp_u_2))],
                  [np.dot(np.matmul(i, u_1), temp_u_2), np.dot(u_1, temp_u_2)]])
    a_b = np.linalg.solve(C, np.array([np.dot(nu_1, nu_2), - np.dot(np.matmul(i, nu_1), nu_2)]))
    return list(a_b)

list_of_matrices = []        # list of transition matrices with different u_1's

for n in range(2):
    list_of_4_matrices_from_the_same_u_1 = []
    while True:  
        try:
            while True:
                u_1 = np.random.randn(d,)
                u_1 = u_1 - np.dot(u_1, nu_1) * nu_1
                u_1 = u_1 - np.dot(u_1, np.matmul(i, nu_1)) * np.matmul(i, nu_1)
                u_1 /= np.linalg.norm(u_1)
                if np.any(u_1):
                    break
            J_0_u_1 = mm(i, u_1)


            while True:
                temp_u_2 = np.random.randn(d,)
                temp_u_2 -= np.dot(temp_u_2, nu_2) * nu_2
                temp_u_2 -= np.dot(temp_u_2, np.matmul(i, nu_2)) * np.matmul(i, nu_2)
                temp_u_2 /= np.linalg.norm(temp_u_2)
                if np.any(temp_u_2):
                    break
            J_0_temp_u_2 = mm(i, temp_u_2)
            print('1')
            get_alpha_beta(u_1, temp_u_2)
            break
        except ValueError:
            pass
        
    print('HHHHEEEERRRRREEEEE')
    print(np.dot(u_1, nu_1))
    print(np.dot(u_1, np.matmul(i, nu_1)))

    alpha_beta_list = get_alpha_beta(u_1, temp_u_2)
    alpha_1 = alpha_beta_list[0]
    beta_1 = alpha_beta_list[1]
    alpha_2 = alpha_beta_list[2]
    beta_2 = alpha_beta_list[3]

    print('1????: {}'.format(alpha_1**2 + beta_1**2))

    list_of_u_2 = []
    list_of_J_0_u_2 = []
    u_2 = alpha_1 * temp_u_2 + beta_1 * np.matmul(i, temp_u_2)
    list_of_u_2.append(u_2)
    list_of_J_0_u_2.append(np.matmul(i,u_2))

    u_2 = alpha_2 * temp_u_2 + beta_2 * np.matmul(i, temp_u_2)
    list_of_u_2.append(u_2)
    list_of_J_0_u_2.append(np.matmul(i,u_2))

    #print('1st u_2: {}'.format(alpha_1 * temp_u_2 + beta_1 * np.matmul(i, temp_u_2)))
    #print('2nd u_2: {}'.format(alpha_2 * temp_u_2 + beta_2 * np.matmul(i, temp_u_2)))


    print('<nu_1,nu_2>: {}'.format(np.dot(nu_1,nu_2)))
    print('1st <u_1,u_2>: {}'.format(np.dot(u_1,alpha_1 * temp_u_2 + beta_1 * np.matmul(i, temp_u_2))))
    print('<i*nu_1,nu_2>: {}'.format(np.dot(np.matmul(i,nu_1),nu_2)))
    print('1st <i*u_1,u_2>: {}'.format(np.dot(np.matmul(i,u_1),alpha_1 * temp_u_2 + beta_1 * np.matmul(i, temp_u_2))))

    print('<nu_1,nu_2>: {}'.format(np.dot(nu_1,nu_2)))
    print('2nd <u_1,u_2>: {}'.format(np.dot(u_1,alpha_2 * temp_u_2 + beta_2 * np.matmul(i, temp_u_2))))
    print('<i*nu_1,nu_2>: {}'.format(np.dot(np.matmul(i,nu_1),nu_2)))
    print('2nd <i*u_1,u_2>: {}'.format(np.dot(np.matmul(i,u_1),alpha_2 * temp_u_2 + beta_2 * np.matmul(i, temp_u_2))))


    while True:
        try:
            while True:
                v_1 = np.random.randn(d,)
                v_1 -= dt(v_1, nu_1) * nu_1
                v_1 -= dt(v_1, mm(i, nu_1)) * mm(i, nu_1)
                v_1 -= dt(v_1, u_1) * u_1
                v_1 -= dt(v_1, mm(i,u_1)) * mm(i,u_1)
                v_1 /= np.linalg.norm(v_1)
                if np.any(v_1):
                    break
            J_0_v_1 = mm(i, v_1)

            while True:
                temp_v_2 = np.random.randn(d,)
                temp_v_2 -= dt(temp_v_2, nu_2) * nu_2
                temp_v_2 -= dt(temp_v_2, mm(i, nu_2)) * mm(i, nu_2)
                temp_v_2 -= dt(temp_v_2, u_2) * u_2
                temp_v_2 -= dt(temp_v_2, mm(i, u_2)) * mm(i, u_2)
                temp_v_2 /= np.linalg.norm(temp_v_2)
                if np.any(temp_v_2):
                    break
            J_0_temp_v_2 = mm(i, temp_v_2)
            print('1')
            get_alpha_beta(v_1, temp_v_2)
            break
        except ValueError:
            pass
    
    print('HERE')
    print(np.dot(v_1, nu_1))
    print(np.dot(v_1, np.matmul(i, nu_1)))
    print(np.dot(v_1, u_1))
    print(np.dot(v_1, np.matmul(i, u_1)))


    alpha_beta_list = get_alpha_beta(v_1, temp_v_2)
    alpha_1 = alpha_beta_list[0]
    beta_1 = alpha_beta_list[1]
    alpha_2 = alpha_beta_list[2]
    beta_2 = alpha_beta_list[3]

    print('1????: {}'.format(alpha_1**2 + beta_1**2))

    list_of_v_2 = []
    list_of_J_0_v_2 = []
    v_2 = alpha_1 * temp_v_2 + beta_1 * np.matmul(i, temp_v_2)
    list_of_v_2.append(v_2)
    list_of_J_0_v_2.append(np.matmul(i,v_2))

    v_2 = alpha_2 * temp_v_2 + beta_2 * np.matmul(i, temp_v_2)
    list_of_v_2.append(v_2)
    list_of_J_0_v_2.append(np.matmul(i,v_2))
    


    #print('1st v_2: {}'.format(alpha_1 * temp_v_2 + beta_1 * np.matmul(i, temp_v_2)))
    #print('2nd v_2: {}'.format(alpha_2 * temp_v_2 + beta_2 * np.matmul(i, temp_v_2)))


    print('<nu_1,nu_2>: {}'.format(np.dot(nu_1,nu_2)))
    print('1st <v_1,v_2>: {}'.format(np.dot(v_1,alpha_1 * temp_v_2 + beta_1 * np.matmul(i, temp_v_2))))
    print('<i*nu_1,nu_2>: {}'.format(np.dot(np.matmul(i,nu_1),nu_2)))
    print('1st <i*v_1,v_2>: {}'.format(np.dot(np.matmul(i,v_1),alpha_1 * temp_v_2 + beta_1 * np.matmul(i, temp_v_2))))

    print('<nu_1,nu_2>: {}'.format(np.dot(nu_1,nu_2)))
    print('2nd <v_1,v_2>: {}'.format(np.dot(v_1,alpha_2 * temp_v_2 + beta_2 * np.matmul(i, temp_v_2))))
    print('<i*nu_1,nu_2>: {}'.format(np.dot(np.matmul(i,nu_1),nu_2)))
    print('2nd <i*v_1,v_2>: {}'.format(np.dot(np.matmul(i,v_1),alpha_2 * temp_v_2 + beta_2 * np.matmul(i, temp_v_2))))

    for u_2 in list_of_u_2: 
        for v_2 in list_of_v_2:
            M = np.concatenate((u_2[:, np.newaxis], v_2[:, np.newaxis], np.matmul(i,u_2)[:, np.newaxis], np.matmul(i,v_2)[:, np.newaxis]), axis=1)
            Q, R = np.linalg.qr(M)

            column_1 = solve_triangular(R, Q.T.dot(xi_to_xi(u_1, nu_1, nu_2)), lower=False)
            column_2 = solve_triangular(R, Q.T.dot(xi_to_xi(v_1, nu_1, nu_2)), lower=False)
            column_3 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_u_1, nu_1, nu_2)), lower=False)
            column_4 = solve_triangular(R, Q.T.dot(xi_to_xi(J_0_v_1, nu_1, nu_2)), lower=False)


            transition_matrix = np.concatenate((column_1[:, np.newaxis], column_2[:, np.newaxis], column_3[:, np.newaxis], column_4[:, np.newaxis]), axis=1)
            #print('transition matrix: \n{}'.format(transition_matrix))

            tr = np.trace(transition_matrix)
            #print('the trace is: {}'.format(tr))

            #print('eigenvalues: {}'.format(np.linalg.eig(transition_matrix)))
            
            while True:
                v = np.random.randn(d-2,)
                if np.any(v):
                    break

            transition_matrix_v = np.matmul(transition_matrix,v)
            print('<v, Av>: {}'.format(np.dot(v,transition_matrix_v)))
            print('<J_0 v, Av>: {}'.format(np.dot(np.matmul(i4,v),np.matmul(transition_matrix,v))))
            print('DET OF A: {}'.format(np.linalg.det(transition_matrix)))
            print('DET OF (A - Id): {}'.format(np.linalg.det(transition_matrix - np.identity(4))))
            print('ORTH? (1)\n{}'.format(np.matmul(transition_matrix, transition_matrix.T)-np.identity(4)))
            print('\n(2)\n{}'.format(np.matmul(transition_matrix.T, transition_matrix)-np.identity(4)))
            print('SYMP? \n{}'.format(np.matmul(np.matmul(np.matmul(transition_matrix.T, np.linalg.inv(i4)), transition_matrix), i4)-np.identity(4)))
            
            list_of_4_matrices_from_the_same_u_1.append(transition_matrix)
    
    list_of_matrices.append(list_of_4_matrices_from_the_same_u_1)


'''random_2_indices_out_of_n = np.random.choice(list(range(n)), 2, replace=False)
random_1_indices_out_of_4 = np.random.choice(list(range(4)), 1, replace=False)
random_2_matrices = [list_of_matrices[random_2_indices_out_of_n[0]][random_1_indices_out_of_4[0]], list_of_matrices[random_2_indices_out_of_n[1]][random_1_indices_out_of_4[0]]]
A = random_2_matrices[0]
B = random_2_matrices[1]'''

'''AB = np.array([[A[0][0]-B[0][0]+A[0][1]+B[1][0]+A[0][2]+B[2][0]+A[0][3]+B[3][0], A[0][1]-B[0][1]-A[0][0]+B[1][1]+A[0][3]+B[2][1]-A[0][2]+B[3][1],
                A[0][2]-B[0][2]-A[0][3]+B[1][2]-A[0][0]+B[2][2]+A[0][1]+B[3][2], A[0][3]-B[0][3]+A[0][2]+B[1][3]-A[0][1]+B[2][3]-A[0][0]+B[3][3]],
               [A[1][0]-B[1][0]+A[1][1]-B[0][0]+A[1][2]+B[3][0]+A[1][3]-B[2][0], A[1][1]-B[1][1]-A[1][0]-B[0][1]+A[1][3]+B[3][1]-A[1][2]-B[2][1],
                A[1][2]-B[1][2]-A[1][3]-B[0][2]-A[1][0]+B[3][2]+A[1][1]-B[2][2], A[1][3]-B[1][3]+A[1][2]-B[0][3]-A[1][1]+B[3][3]-A[1][0]-B[2][3]],
               [A[2][0]-B[2][0]+A[2][1]-B[3][0]+A[2][2]-B[0][0]+A[2][3]+B[1][0], A[2][1]-B[2][1]-A[2][0]-B[3][1]+A[2][3]-B[0][1]-A[2][2]+B[1][1],
                A[2][2]-B[2][2]-A[2][3]-B[3][2]-A[2][0]-B[0][2]+A[2][1]+B[1][2], A[2][3]-B[2][3]+A[2][2]-B[3][3]-A[2][1]-B[0][3]-A[2][0]+B[1][3]],
               [A[3][0]-B[3][0]+A[3][1]+B[2][0]+A[3][2]-B[1][0]+A[3][3]-B[0][0], A[3][1]-B[3][1]-A[3][0]+B[2][1]+A[3][3]-B[1][1]-A[3][2]-B[0][1],
                A[3][2]-B[3][2]-A[3][3]+B[2][2]-A[3][0]-B[1][2]+A[3][1]-B[0][2], A[3][3]-B[3][3]+A[3][2]+B[2][3]-A[3][1]-B[1][3]-A[3][0]-B[0][3]]])
x = np.linalg.solve(AB, np.zeros(4,))
print(x)'''


'''for l in range(4):
    A = list_of_matrices[0][l]
    for m in range(4):
        B = list_of_matrices[1][m]
        difference_matrix = np.matmul(B, np.linalg.inv(A))

        print('difference matrix: \n{}'.format(difference_matrix))
        print('det: {}'.format(np.linalg.det(difference_matrix)))
        print('ORTH? (1)\n{}'.format(np.matmul(difference_matrix, difference_matrix.T)-np.identity(4)))
        print('(2)\n{}'.format(np.matmul(difference_matrix.T, difference_matrix)-np.identity(4)))
        print('SYMP? \n{}'.format(np.matmul(np.matmul(np.matmul(difference_matrix.T, np.linalg.inv(i4)), difference_matrix), i4)-np.identity(4)))
'''
