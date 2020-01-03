import numpy as np


def vec(matrix):
    return matrix.reshape(-1, 1)


def mat(vector, N):
    return vector.reshape(-1, N)


def join(a, b):
    return np.concatenate([a, b])


def normalize(v, s=1):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v * np.sqrt(s) / norm


def fast_inverse(A_inverse, x):
    xxT = np.outer(x, x)
    d1 = A_inverse.dot(xxT).dot(A_inverse)
    d2 = x.dot(A_inverse).dot(x.T) + 1
    next_A_inverse = A_inverse - d1/d2
    return next_A_inverse
