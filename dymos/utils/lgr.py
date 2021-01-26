import numpy as np


def lgr(n, include_endpoint=False, tol=1.0E-15):
    """
    Returns the Legendre-Gauss-Radau nodes and weights for a Jacobi Polynomial with n abscissae.

    Parameters
    ----------
    n : int
        The number of LGR nodes are to be returned.  If include_endpoint=True we return n+1 values.
    include_endpoint : bool
        If True return the non-free abscissa at the right endpoint of the interval [-1, 1].
        The weight associated with this endpoint is 0.
    tol : float
        The tolerance to which the location of the nodes should be converged.

    Returns
    -------
    numpy.array
        An array of the LGR nodes for a polynomial of the given order.
    numpy.array
        An array of the corresponding LGR weights at the nodes in x.

    Notes
    -----
    The LGR nodes are the roots of

        .. math::
            (P_N(x)+P_{N+1}(x))/(x+1).

    References
    ----------
    C. Canuto, M. Y. Hussaini, A. Quarteroni, T. A. Tang, "Spectral Methods
    in Fluid Dynamics," Section 2.3. Springer-Verlag 1987

    F. B. Hildebrand , "Introduction to Numerical Analysis," Section 8.11
    Dover 1987

    The nodes are on the range [-1, 1).

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
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
    OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    n1 = n
    n = n - 1

    # The Chebyshev-Gauss-Radau nodes serve as the initial guess.
    x = -np.cos(2.0 * np.pi * np.arange(0, n1) / (2 * n + 1))

    # The Legendre vandermonde Matrix
    P = np.zeros((n1, n1 + 1))

    # Compute P_(N) using the recursion relation
    # Copute its first and second derivatives and
    # update x using the Newton-Raphson method.
    xold = np.ones_like(x)

    # Free abscissae
    free = np.arange(1, n1)

    for i in range(100):
        if np.all(np.abs(x - xold) <= tol):
            break

        xold[:] = x

        P[0, :] = np.power(-1.0, np.arange(0, n1 + 1))
        P[free, 0] = 1.0
        P[free, 1] = x[free]

        for k in range(1, n1):
            P[free, k + 1] = (
                (2 * (k + 1) - 1) * x[free] * P[free, k] -
                ((k + 1) - 1) * P[free, k - 1]
            ) / (k + 1)

        f = ((1.0 - xold[free]) / n1) * (P[free, n] + P[free, n1])
        fprime = (P[free, n] - P[free, n1])

        x[free] = xold[free] - f / fprime

    else:
        raise RuntimeError('Failed to converge LGR nodes for order {0}'.format(n))

    # Compute the weights
    w = np.zeros(n1)
    w[0] = 2.0 / n1**2
    w[free] = (1.0 - x[free]) / (n1 * P[free, n])**2

    # Tack on the endpoint if requested
    if include_endpoint:
        x = np.concatenate([x, [1.0]])
        w = np.concatenate([w, [0.0]])

    return x, w
