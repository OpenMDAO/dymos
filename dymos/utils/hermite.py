import numpy as np


def hermite_matrices(x_given, x_eval):
    """
    Return matrices for a Hermite polynomial at the given nodes.

    This includes interpolation matrices (A_i and B_i) and differentiation matrices (A_d and B_d).

    Parameters
    ----------
    x_given : ndarray[:]
        Vector of given nodes in the polynomial.
    x_eval : ndarray[:]
        Vector of nodes at which the polynomial is evaluated.

    Returns
    -------
    A_i : np.array
        An num_disc_nodes-1 x num_disc_nodes matrix used for the interpolation of state values
        at the interior LGL nodes.
    B_i : np.array
        An num_disc_nodes-1 x num_disc_nodes matrix used for the interpolation of state values
        at the interior LGL nodes.
    A_d : np.array
        An num_disc_nodes-1 x num_disc_nodes matrix used for the differentiation of state values
        at the collocation LGL nodes.
    B_d : np.array
        An num_disc_nodes-1 x num_disc_nodes matrix used for the differentiation of state values
        at the collocation LGL nodes.

    Notes
    -----
    .. math::
        x_i = \\left[ A_i\\right] x_c + \\frac{dt}{dtau} \\left[ B_i \\right] f_c
    """
    tau_disc = x_given
    """ Array of the discretization nodes. """

    num_disc_nodes = len(x_given)
    """ Number of discretization nodes per segment. """

    tau_col = x_eval
    """ Array of the collocation nodes. """

    num_col_nodes = len(x_eval)
    """ Number of collocation nodes per segment. """

    Ai = np.zeros([num_col_nodes, num_disc_nodes])
    Bi = np.zeros([num_col_nodes, num_disc_nodes])
    Ad = np.zeros([num_col_nodes, num_disc_nodes])
    Bd = np.zeros([num_col_nodes, num_disc_nodes])

    # The state interpolation matrices
    for i in range(num_col_nodes):
        ui, vi = heriwi(tau_col[i], tau_disc)
        Ai[i, :] = ui
        Bi[i, :] = vi

    # The differentiation matrices
    for i in range(num_col_nodes):
        ui, vi = heriwd(tau_col[i], tau_disc)
        Ad[i, :] = ui
        Bd[i, :] = vi

    return Ai, Bi, Ad, Bd


def heriwi(tau, taus):
    """
    Computes the weights for computing the values of polynomial with hermite interpolation.

    Based off of the ACM211 algorithm.

    Parameters
    ----------
    tau : float
        Value at which the Hermite polynomial weights are desired.
    taus : np.array
        Array of points at which the values and derivatives which
        define the Hermite polynomial are provided.

    Returns
    -------
    u : np.array
        Weights for function values.
    v : np.array
        Weights for derivative values.
    """
    n = len(taus)
    u = np.zeros(n)
    v = np.zeros(n)

    for j in range(n):
        prod = 1.0
        sum1 = 0.0
        for i in range(n):
            if i != j:
                prod *= ((tau - taus[i]) / (taus[j] - taus[i]))**2
                sum1 += 1. / (taus[j] - taus[i])
        u[j] = prod * ((taus[j] - tau) * 2.0 * sum1 + 1.0)
        v[j] = prod * (tau - taus[j])

    return u, v


def heriwd(tau, taus):
    """
    Computes the weights for computing the derivatives of a polynomial with hermite interpolation.

    Based off of the ACM211 algorithm

    Parameters
    ----------
    tau : float
        Value at which the Hermite polynomial weights are desired.
    taus : np.array
        Array of points at which the values and derivatives which
        define the Hermite polynomial are provided.

    Returns
    -------
    u : np.array
        Weights for function values.
    v : np.array
        Weights for derivative values.
    """
    n = len(taus)
    u = np.zeros(n)
    v = np.zeros(n)

    for j in range(n):
        prod = 1.0
        dprod = 0.0
        sum1 = 0.0
        for i in range(n):
            if i != j:
                xmxi = tau - taus[i]
                xjmxi = taus[j] - taus[i]
                dprod = dprod * (xmxi / xjmxi)**2 + 2.0 * prod * xmxi / xjmxi**2
                prod = prod * (xmxi / xjmxi)**2
                sum1 = sum1 + 1.0 / xjmxi
        xmxj = tau - taus[j]
        xjmx = taus[j] - tau
        u[j] = dprod * (xjmx * 2.0 * sum1 + 1.0) - prod * (2.0 * sum1)
        v[j] = dprod * xmxj + prod

    return u, v
