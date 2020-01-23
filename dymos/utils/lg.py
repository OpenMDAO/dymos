import numpy as np


def lg(n, tol=1.0E-15):
    """
    Returns the Legendre-Gauss nodes and weights for a
    Jacobi Polynomial with n abscissae.

    The nodes are on the range (-1, 1).

    Based on the routine written by Greg von Winckel (License follows)

    Copyright (c) 2004, Greg von Winckel
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
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    n : int
        The number of Legendre-Gauss nodes to be returned
    tol : float
        The tolerance to which the location of the nodes should be converged.

    Returns
    -------
    x : numpy.array
        An array of the LG nodes for a polynomial of the given order.

    w : numpy.array
        An array of the corresponding LG weights at the nodes in x.

    """
    n = n - 1
    n1 = n + 1
    n2 = n + 2

    x0 = 2.0 * np.ones(n1)
    xu = np.linspace(-1, 1, n1)

    L = np.zeros((n1, n2))

    # Chebyshev nodes as a guess
    x = np.cos(np.pi * (2.0 * np.linspace(1, n1, n1) - 1.0) / (2.0 * n1))
    x += (0.27 / n1) * np.sin(np.pi * xu * n / n2)

    L[:, 0] = 1.0

    for i in range(100):
        if np.all(np.abs(x - x0)) <= tol:
            break

        L[:, 1] = x

        for k in range(2, n2):
            L[:, k] = ((2 * k - 1) * x * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k

        Lp = n2 * (L[:, n] - x * L[:, n1]) / (1 - x**2)

        x0 = x
        x = x0 - L[:, n1] / Lp
    else:
        raise RuntimeError('Failed to converge LG nodes for order {0}'.format(n))

    # Put the nodes in ascending order
    x = -x

    # Compute the weights
    w = 2.0 / ((1.0 - x**2) * Lp**2) * (n2 / n1)**2

    return x, w
