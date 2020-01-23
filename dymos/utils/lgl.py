import numpy as np

_lgl_cache = {}
""" Cache for the LGL nodes and weights, keyed by n. """


def _lgl(n, tol=np.finfo(float).eps):
    """
    Returns the Legendre-Gauss-Lobatto nodes and weights for a
    Jacobi Polynomial with n abscissae.

    The nodes are on the range [-1, 1].

    Based on the routine written by Greg von Winckel (License follows)

    Copyright (c) 2009, Greg von Winckel
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the distribution

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR aux_outputs
    PARTICULAR PURPOSE  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    n : int
        The number of LGL nodes requested.  The order of the polynomial is n-1.
    tol : float
        The tolerance to which the location of the nodes should be converged.

    Returns
    -------
    x : numpy.array
        An array of the LGL nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding LGL weights at the nodes in x.

    """
    n = n - 1
    n1 = n + 1
    n2 = n + 2
    # Get the initial guesses from the Chebyshev nodes
    x = np.cos(np.pi * (2 * np.arange(1, n2) - 1) / (2 * n1))
    P = np.zeros([n1, n1])
    # Compute P_(n) using the recursion relation
    # Compute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = 2

    for i in range(100):
        if np.all(np.abs(x - xold) <= tol):
            break
        xold = x
        P[:, 0] = 1.0
        P[:, 1] = x

        for k in range(2, n1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

        x = xold - (x * P[:, n] - P[:, n - 1]) / (n1 * P[:, n])
    else:
        raise RuntimeError('Failed to converge LGL nodes '
                           'for order {0}'.format(n))

    x.sort()

    w = 2 / (n * n1 * P[:, n]**2)

    return x, w


def lgl(n):
    """ Retrieve the lgl nodes and weights for n nodes.

    Results are cached to avoid repeated calculation of nodes and weights for a given n.
    """
    if n not in _lgl_cache:
        _lgl_cache[n] = _lgl(n)
    return _lgl_cache[n]
