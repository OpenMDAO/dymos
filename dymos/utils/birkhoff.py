import numpy as np
import scipy.special as sp


def birkhoff_matrix(tau, w, grid_type):
    """
    Returns the pseudospectral integration matrix for a Birkhoff polynomial at the given nodes.

    Parameters
    ----------
    tau : ndarray[:]
        Vector of given nodes in the polynomial.
    w : ndarray[:]
        Vector of nodes at which the polynomial is evaluated.
    grid_type : str
        The type of Gaussian grid used in the transcription.

    Returns
    -------
    np.array
        An num_disc_nodes-1 x num_disc_nodes matrix used for the interpolation of state values
        at the interior LGL nodes.
    """
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
        for i in range(1, N + 1):  # The first row is exactly zero.
            S[i, 0] = (tau[i] - tau[0]) / 2

            for n in range(1, N+1):
                gamma = 2 / (2 * n + 1)
                int_p = (pol(n+1, tau[i]) - pol(n-1, tau[i])) / (2*n+1)
                S[i, n] = int_p / gamma

    elif grid_type[0] == 'c':
        gamma = np.pi / 2
        for i in range(1, N+1):  # The first row is exactly zero.
            # chebyshev polynomial of order 0: 1
            S[i, 0] = (tau[i] - tau[0]) / np.pi

            # chebyshev polynomial of order 1: x
            S[i, 1] = (tau[i]**2 - tau[0]**2) / np.pi

            for n in range(2, N+1):
                int_p = pol(n+1, tau[i]) / (2*n+2) - pol(n-1, tau[i]) / (2*n-2) - (-1)**n / (n**2 - 1)
                S[i, n] = int_p / gamma

    B = S @ alpha

    return B
