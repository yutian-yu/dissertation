import numpy as np
import math as math
import random

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

A = np.array([[2, 3, 1, 4],
              [0, 6, 4, 2],
              [0, 0, 3, 4],
              [0, 0, 0, 1]])

print(np.matmul(A, i4))
print(np.matmul(i4, A))
