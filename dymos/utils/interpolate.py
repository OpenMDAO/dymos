import numpy as np


class LagrangeBarycentricInterpolant(object):
    """
    LagrangeBarycentricInterpolant interpolates values and first derivatives
    of set of data using barycentric interpolation of a Lagrange Polynomial.

    Parameters
    ----------
    nodes : sequence
        The nodes of the polynomial from -1 to 1 on which the values to
        be interpolated are given.

    Attributes
    ----------
    num_nodes : int
        The number of nodes in the interpolated polynomial.
    tau_i : np.array
        The location of the nodes of the polynomial in tau space [-1 1]
    x_0 : float
        The x-value corresponding to the left-side of the interval (tau = -1)
    x_f : float
        The x-value corresponding to the right-side of the interval (tau = +1)
    f_j : np.array
        The values to be interpolated
    w_b : np.array
        The barycentric weights for the points in the interpolated polynomial
    wbfj : np.array
        An array of the precomputed product of the interpolated values and
        the corresponding barycentric weights.
    dx_dtau : float
        Half the span from x0 to xf.  The ratio of x-space to
        internal tau-space.

    Notes
    -----
    The Barycentric formula is given in Eq. 3.3 of [1]_ as

    .. math::

        p(x) = l(x) \\Sum \\frac{w_j f_j}{x-x_j}

    where l(x) is

    .. math::

        l(x) = (x-x_0)(x-x_1)(x-x_2)...

    The singularity in the denominator of p(x) at x = x_n is cancelled
    by the the term (x-x_n) in l(x).

    References
    ----------
    .. [1] Berrut, Jean-Paul, and Lloyd N. Trefethen.
       "Barycentric lagrange interpolation." Siam Review 46.3 (2004): 501-517.

    """

    def __init__(self, nodes, shape):

        self.num_nodes = len(nodes)
        """ The number of nodes in the interpolated polynomial. """

        self.tau_i = nodes
        """ The independent variable values at interpolation points. """

        _shape = (self.num_nodes,) + shape

        self.f_j = np.zeros(_shape)
        """ An array of values to be interpolated. """

        self.w_b = np.ones(self.num_nodes)
        """ Barycentric weights for nodes in the interpolated polynomial."""

        # Barycentric Weights
        for j in range(self.num_nodes):
            for k in range(self.num_nodes):
                if k != j:
                    self.w_b[j] /= (self.tau_i[j] - self.tau_i[k])

        self.wbfj = np.zeros(_shape)
        """ An array of the precomputed product of the interpolated
            values and the corresponding barycentric weights.
        """

        n = self.wbfj.shape[0]
        m = np.prod(self.wbfj.shape[1:])
        self.wbfj_flat = np.reshape(self.wbfj, newshape=(n, m))
        """ A flattened view of wbfj"""

        self.x0 = -1.0
        """ The value of the independent axis corresponding to $\tau = -1$ """

        self.xf = 1.0
        """ The value of the independent axis corresponding to $\tau = 1$ """

        self.dx_dtau = 1.0
        """ Half the span from x0 to xf.  The ratio of x-space to
        internal tau-space. """

        self._is_setup = False

    def x_to_tau(self, x):
        """ Converts the independent variable x to its corresponding
        value of $\tau$.

        Given bounds on the independent variable x0 and xf which
        correspond to $\tau$ of -1 and 1, respectively, the returned value
        will be the equivalent $tau$.

        For instance, if x0 = 0 and xf = 100, x = 50 will have an equivalent
        $\tau$ of 0 (halfway on [-1 1]).

        :param x: The independent variable to be converted to $\tau$.
        :return: The equivalent value of $\tau$ given x.

        """
        return -1.0 + (x - self.x0) / self.dx_dtau

    def setup(self, x0, xf, f_j):
        """Prepare the interpolant for use by setting the
        values to be interpolated.

        Parameters
        ----------
        x0 : float
            The lower bound of the independent variable,
            corresponding to $\tau = -1$
        xf : float
            The upper bound of the independent variable,
            corresponding to $\tau = -1$
        f_j : np.array
            The values to be interpolated at the LGL nodes on [-1 1]

        Raises
        ------
        ValueError
            If the length of f_j is not the number of nodes in the interpolated
            polynomial.

        """
        if len(f_j) != self.num_nodes:
            raise ValueError("f_j must have {0} values".format(self.num_nodes))
        self.f_j[...] = f_j
        self.x0 = x0
        self.xf = xf
        self.dx_dtau = 0.5 * (xf - x0)

        fjT = self.f_j.T
        self.wbfj[...] = (self.w_b * fjT).T
        self._is_setup = True

    def eval(self, x):
        """ Interpolate the LGL polynomial at x.

        Parameters
        ----------
        x : float
            The independent variable value at which interpolation
            is requested.

        Returns
        -------
        float
            The interpolated value of the LGL polynomial at x.

        """
        if not self._is_setup:
            raise RuntimeError('LagrangeBarycentricInterpolant has not been setup')
        tau = self.x_to_tau(x)

        g = tau - self.tau_i
        l = np.ones_like(g)

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if j == i:
                    continue
                l[i] *= g[j]

        result = np.reshape(np.dot(l, self.wbfj_flat), newshape=self.wbfj.shape[1:])

        return result

    def eval_deriv(self, x, der=1):
        """ Interpolate the derivative of the polynomial at x.

        Parameters
        ----------
        x : float
            The independent variable value at which the derivative
            is requested.

        Returns
        -------
        float
            The first derivative of the polynomial at x.

        """
        if not self._is_setup:
            raise RuntimeError('LagrangeBarycentricInterpolant has not been setup')
        if der >= self.num_nodes:
            return 0.0

        n = self.num_nodes
        tau = self.x_to_tau(x)
        g = tau - self.tau_i
        lprime = np.zeros(n)

        if der == 1:
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    prod = 1.0
                    for k in range(n):
                        if k in {i, j}:
                            continue
                        prod *= g[k]
                    lprime[i] += prod
            # df_dtau = np.dot(lprime, self.wbfj)
            df_dtau = np.reshape(np.dot(lprime, self.wbfj_flat), newshape=self.wbfj.shape[1:])
            return df_dtau / self.dx_dtau
        elif der == 2:
            for i in range(n):
                for j in range(n):
                    if j == i:
                        continue
                    for k in range(n):
                        if k in {i, j}:
                            continue
                        prod = 1.0
                        for ii in range(n):
                            if ii in {i, j, k}:
                                continue
                            prod *= g[ii]
                        lprime[i] += prod
            df_dtau = np.reshape(np.dot(lprime, self.wbfj_flat), newshape=self.wbfj.shape[1:])
            return df_dtau / self.dx_dtau**2
        else:
            raise ValueError('Barycentric interpolant currently only supports up to '
                             'second derivatives')
