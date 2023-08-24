import numpy as np
import scipy.special as sp


def birkhoff_matrices(tau, w):
    N = tau.size - 1
    B = np.zeros((N + 1, N + 1))

    is_gl = True if tau[-1] == 1 else False
    end_node = N if is_gl else N + 1

    alpha = np.zeros((N + 1, N + 1))

    for j in range(0, N + 1):
        for n in range(0, end_node):
            alpha[n, j] = w[j] * sp.eval_legendre(n, tau[j])
        if is_gl:
            alpha[N, j] = N * sp.eval_legendre(N, tau[j]) * w[j] / (2 * N + 1)

    for i in range(0, N + 1):
        for j in range(0, N + 1):
            s = alpha[0, j] * (tau[i] - tau[0]) / 2

            for n in range(1, N):
                gamma = 2 / (2 * n + 1)
                int_p = (sp.eval_legendre(n + 1, tau[i]) - sp.eval_legendre(n - 1, tau[i])) / (2 * n + 1)

                s += alpha[n, j] * int_p / gamma

            B[i, j] = s

    return B
