import numpy as np


def lagrange_matrices(x_disc, x_interp):
    """
    Compute the lagrange matrices.

    The lagrange matrics are given 'discretization' nodes at which
    values are specified and 'interpolation' nodes at which values will be desired,
    returns interpolation and differentiation matrices which provide polynomial
    values and derivatives.

    Parameters
    ----------
    x_disc : np.array
        The cardinal nodes at which values of the variable are specified.
    x_interp : np.array
        The interior nodes at which interpolated values of the variable or its derivative
        are desired.

    Returns
    -------
    np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at the cardinal nodes, returns the intepolated values at the interior
        nodes.

    np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at the cardinal nodes, returns the intepolated derivatives at the interior
        nodes.
    """
    nd = len(x_disc)
    ni = len(x_interp)

    Li = np.zeros((ni, nd))
    Di = np.zeros((ni, nd))
    temp = np.zeros((ni, nd))

    # Barycentric Weights
    diff = np.reshape(x_disc, (nd, 1)) - np.reshape(x_disc, (1, nd))
    np.fill_diagonal(diff, 1.0)
    wb = np.prod(1.0 / diff, axis=1)

    # Compute Li
    diff = np.reshape(x_interp, (ni, 1)) - np.reshape(x_disc, (1, nd))
    for j in range(nd):
        temp[:] = diff[:]
        temp[:, j] = 1.0
        Li[:, j] = wb[j] * np.prod(temp, axis=1)

    # Compute Di
    for j in range(nd):
        for k in range(nd):
            if k != j:
                temp[:] = diff[:]
                temp[:, j] = 1.0
                temp[:, k] = 1.0
                Di[:, j] += wb[j] * np.prod(temp, axis=1)

    return Li, Di
