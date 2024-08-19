import numpy as np

_cgl_cache = {}
""" Cache for the LGL nodes and weights, keyed by n. """


def clenshaw_curtis(n):
    """
    Returns the Chebyshev-Gauss-Lobatto nodes and weights for a Jacobi Polynomial with n abscissae.

    The nodes are on the range [-1, 1].

    Parameters
    ----------
    n : int
        The number of LGL nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the LGL nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding CGL weights at the nodes in x.
    """
    x = -np.cos(np.pi * np.arange(n) / (n-1))
    w = np.zeros(n)
    even = False if n % 2 else True

    N_2 = int(n/2) if even else int((n+1)/2)
    j_lim = N_2 if even else int((n-1)/2)

    for k in range(n):
        c = 1 if k == 0 else 2
        s = 0
        for j in range(1, N_2):
            b = 1 if j == j_lim else 2
            s += b * np.cos(2*j*k*np.pi/(n-1)) / (4*j**2 - 1)
        w[k] = c * (1 - s) / (n-1)

    if even:
        w[N_2:] = np.flip(w[:N_2])
    else:
        w[N_2:] = np.flip(w[:(N_2-1)])

    return x, w


def _cgl(n):
    """
    Returns the Chebyshev-Gauss-Lobatto nodes and weights for a Jacobi Polynomial with n abscissae.

    The nodes are on the range [-1, 1].

    Parameters
    ----------
    n : int
        The number of CGL nodes requested.  The order of the polynomial is n-1.

    Returns
    -------
    x : numpy.array
        An array of the CGL nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding CGL weights at the nodes in x.
    """
    x = -np.cos(np.pi * np.arange(n) / (n-1))
    w = np.pi / (n-1) * np.ones(n)
    w[0] *= 0.5
    w[-1] *= 0.5

    return x, w


def cgl(n):
    """
    Retrieve the cgl nodes and weights for n nodes.

    Results are cached to avoid repeated calculation of nodes and weights for a given n.

    Parameters
    ----------
    n : int
        Node number.

    Returns
    -------
    float
        Tuple with cgl nodes and weights.
    """
    if n not in _cgl_cache:
        _cgl_cache[n] = _cgl(n)
    return _cgl_cache[n]
