import numpy as np
import math
import random
from sys import float_info

def HPNMF(A, S, max_iter, k, lambda, gama, alpha, beta):
    # A--Adjacency matrix
    # S--Similarity matrix
    # k--Number of communities
    eps = float_info.epsilon
    n = A.shape[0]
    D = np.diagflat(np.sum(S, axis=1))
    A2 = A + alpha * np.eye(n)
    M = np.ones((k, k))
    N = (1 - beta) * np.ones((n, k))
    U = 5.0 * np.random.rand(n, k)

    for niter in range(1, max_iter):
        # updata U
        phi = 2.0 * (alpha + 1) * np.dot(U, np.dot(U.T, U))
        up = 2.0 * np.dot(A2, U) + lambda * np.dot(S, U)
        down = phi + lambda * np.dot(D, U) + gama * np.dot(U, M)
        U = U * (N + beta * up / np.maximum(down, eps))

    return U
