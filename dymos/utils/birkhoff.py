import numpy as np
import scipy.special as sp


def birkhoff_matrices(tau, w):
    N = tau.size - 1
    B = np.zeros((N + 1, N + 1))
    B[:, 0] = 1
    Bd = np.eye(N + 1)

    alpha = np.zeros((N, N+1))
    for j in range(1, N + 1):
        for k in range(0, N):
            alpha[k, j] = w[j] * (sp.eval_legendre(k, tau[j])
                                  - sp.eval_legendre(N, tau[j])
                                  * sp.eval_legendre(k, tau[0])/sp.eval_legendre(N, tau[0]))

    for i in range(0, N + 1):
        # print(f'i = {i}')
        for j in range(1, N + 1):
            # print(f'j = {j}')
            # print(f'k = 0')
            s = alpha[0, j] * (tau[i] - tau[0]) / 2
            # print(alpha[0, j])
            # print((tau[j] - tau[0]))
            # print(s)

            for k in range(1, N):
                # print(f'k = {k}')
                gamma = 2/(2*k + 1)
                int_p = (sp.eval_legendre(k+1, tau[i]) - sp.eval_legendre(k-1, tau[i]) +
                         sp.eval_legendre(k-1, tau[0]) - sp.eval_legendre(k+1, tau[0])) / (2*k+1)

                s += alpha[k, j] * int_p / gamma

                # print(s)

            #     print(alpha[k, j] * int_p / gamma)
            #     print(s)
            #     print('----------')
            #
            # print('----------')

            B[i, j] = s

    sd = 0
    for i in range(0, N + 1):
        for j in range(0, N + 1):
            for k in range(0, N):
                gamma = 2/(2*k + 1)
                sd += alpha[k, j] * (sp.eval_legendre(k, tau[i]) / gamma)
            Bd[i, j] = sd
            sd = 0

    return B, Bd
