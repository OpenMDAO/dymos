import numpy as np
import scipy.special as sp


def birkhoff_matrices(tau, w, grid_type):
    N = tau.size - 1
    end_node = N if tau[-1] == 1 else N + 1

    alpha = np.zeros((N + 1, N + 1))
    S = np.zeros((N + 1, N + 1))

    if grid_type[0] == 'l':
        pol = sp.eval_legendre
    elif grid_type[0] == 'c':
        pol = sp.eval_chebyt
    else:
        raise ValueError('invalid grid type')

    for j in range(0, N + 1):
        for n in range(0, end_node):
            alpha[n, j] = w[j] * pol(n, tau[j])
        if grid_type == 'lgl':
            alpha[N, j] = N * pol(N, tau[j]) * w[j] / (2 * N + 1)
        elif grid_type == 'cgl':
            alpha[N, j] = pol(N, tau[j]) * w[j] / 2

    if grid_type[0] == 'l':
        for i in range(0, N + 1):
            S[i, 0] = (tau[i] - tau[0]) / 2

            for n in range(1, N):
                gamma = 2 / (2 * n + 1)
                int_p = (pol(n+1, tau[i]) - pol(n-1, tau[i])) / (2*n+1)
                S[i, n] = int_p / gamma

    elif grid_type[0] == 'c':
        gamma = np.pi / 2
        for i in range(0, N+1):
            S[i, 0] = (tau[i] - tau[0]) / np.pi
            S[i, 1] = (tau[i]**2 - tau[0]**2) / np.pi

            for n in range(2, N):
                int_p = pol(n+1, tau[i]) / (2*n+2) - pol(n-1, tau[i]) / (2*n-2) - (-1)**n / (n**2 - 1)
                S[i, n] = int_p / gamma

    B = S @ alpha

    return B
