import numpy as np


def lagrange_matrices(x_disc, x_interp):
    """
    Compute the lagrange matrices which, given 'cardinal' nodes at which
    values are specified and 'interior' nodes at which values will be desired,
    returns interpolation and differentiation matrices which provide polynomial
    values and derivatives

    Parameters
    ----------
    x_disc : np.array
        The cardinal nodes at which values of the variable are specified.
    x_interp : np.array
        The interior nodes at which interpolated values of the variable or its derivative
        are desired.

    Returns
    -------
    Li : np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at the cardinal nodes, yields the intepolated values at the interior
        nodes.

    Di : np.array
        A num_i x num_c matrix which, when post-multiplied by values specified
        at the cardinal nodes, yields the intepolated derivatives at the interior
        nodes.

    """
    nd = len(x_disc)
    ni = len(x_interp)
    wb = np.ones(nd)

    Li = np.zeros((ni, nd))
    Di = np.zeros((ni, nd))

    # Barycentric Weights
    for j in range(nd):
        for k in range(nd):
            if k != j:
                wb[j] /= (x_disc[j] - x_disc[k])

    # Compute Li
    for i in range(ni):
        for j in range(nd):
            Li[i, j] = wb[j]
            for k in range(nd):
                if k != j:
                    Li[i, j] *= (x_interp[i] - x_disc[k])

    # Compute Di
    for i in range(ni):
        for j in range(nd):
            for k in range(nd):
                prod = 1.0
                if k != j:
                    for m in range(nd):
                        if m != j and m != k:
                            prod *= (x_interp[i] - x_disc[m])
                    Di[i, j] += (wb[j] * prod)

    return Li, Di
