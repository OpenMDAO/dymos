import numpy as np


class SimpleODE:
    """
    An extremely simple ODE that provides the speed of an object falling in a constant gravity field.
    """
    @staticmethod
    def eval(x, t):
        return x - t**2 + 1

    @staticmethod
    def dx(x, t):
        """
        The derivative of the rate of x wrt x.
        """
        return np.array([[1.0]])

    @staticmethod
    def dt(x, t):
        """
        The derivative of the rate of x wrt t.
        """
        return np.array([[-2.0 * t]])

    @staticmethod
    def check(x0, t0, td):
        """
        Compute the analytic solution given the initial conditions and duration of integration.
        """
        tf = t0 + td
        xf = tf**2 + 2 * tf + 1 - x0 * np.exp(tf)
        dx = np.exp(tf)
        dtf = 2 * tf + 2 - x0
        return xf, dx, dtf


# def euler(x0, t0, td, N):
#     """
#     Euler's method with forward-computed derivatives.
#
#     Parameters
#     ----------
#     x0 : np.array
#         The initial state vector.
#     t0 : float
#         The initial time.
#     td : float
#         The time duration of the integration.
#     N : float
#         The number of equally-spaced step-sizes to be taken using Euler's method.
#
#     Returns
#     -------
#     np.array
#         The final state vector after N steps over td units of time.
#     np.array
#         The derivative of the final state vector wrt the initial state vector
#     """
#     if N > 0:
#         h = td/N
#     else:
#         h = 0
#     x = x0.copy()
#     t = t0
#     dx = np.eye(x.shape[0], dtype=complex)
#     print(x, dx)
#
#     for i in range(N):
#
#         # foo = dx_dx * np.eye(x.shape[0], dtype=complex) + h * SimpleODE.dx(x, t)
#         dx = dx_dx @ np.eye(x.shape[0], dtype=complex) + h * SimpleODE.dx(x, t)
#         # print(foo)
#         # print(bar)
#         #
#         # dx_dx *= np.eye(x.shape[0], dtype=complex) + h * SimpleODE.dx(x, t)
#         x += h * SimpleODE.eval(x, t)
#         print(x, dx, h)
#         t += h
#
#     return x, dx_dx


def euler_rev(x0, td, N):
    if N > 0:
        h = td/N
    else:
        h = 0
    t0 = 0
    x = x0.copy()
    t = t0
    x_stack = []
    t_stack = []
    print(x, x_stack)

    x_stack.append(x.copy())
    t_stack.append(t)

    # Compute
    for i in range(N):
        # dx_dx *= np.eye(x.shape[0], dtype=complex) + h * df_ode(x, t)
        x += h * SimpleODE.eval(x, t)
        x_stack.append(x.copy())
        print(x, x_stack)
        t += h
        t_stack.append(t)

    # Deriv
    x_bar = np.eye(x.shape[0], dtype=complex)
    for i in range(N):
        x_i = x_stack.pop()
        t_i = t_stack.pop()
        x_bar = x_bar @ (np.eye(x.shape[0], dtype=complex) + h * SimpleODE.dx(x_i, t_i)).T
        # dx_dx *= np.eye(x.shape[0], dtype=complex) + h * df_ode(x, t)
        # x += h * f_ode(x, t)
        # t += h

    return x, x_bar


def euler(x0, t0, td, N):
    """
    Euler's method with forward-computed derivatives.

    Parameters
    ----------
    x0 : np.array
        The initial state vector.
    t0 : float
        The initial time.
    td : float
        The time duration of the integration.
    N : float
        The number of equally-spaced step-sizes to be taken using Euler's method.

    Returns
    -------
    np.array
        The final state vector after N steps over td units of time.
    np.array
        The derivative of the final state vector wrt the initial state vector
    """
    if N > 0:
        h = td / N
    else:
        h = 0
    x = x0.copy()
    x_size = np.prod(x.shape, dtype=int)
    t = t0

    dx_dx0 = np.eye(x_size, dtype=complex)
    dt_dtd = np.zeros((1, 1), dtype=complex)
    dx_dtd = np.zeros((x_size, 1), dtype=complex)
    dt_dt0 = np.ones((1, 1), dtype=complex)
    dx_dt0 = np.zeros((x_size, 1), dtype=complex)

    dh_dt0 = np.zeros((1, 1), dtype=complex) / N
    dh_dtd = np.ones((1, 1), dtype=complex) / N
    dt_dt = np.ones((1, 1), dtype=complex)
    dt_dh = np.ones((1, 1), dtype=complex)

    I_x = np.eye(x_size, dtype=complex)
    I_t = np.eye(1, dtype=complex)

    for i in range(N):
        f = SimpleODE.eval(x, t)
        f_x = SimpleODE.dx(x, t)
        f_t = SimpleODE.dt(x, t)

        px_px = I_x + h * f_x
        px_pt = h * f_t
        px_ph = f
        pt_pt = I_t

        # Compute this with the initial values of dx_dx and dt_dtd before they're updated
        dx_dtd = px_px @ dx_dtd + \
                 px_pt @ dt_dtd + \
                 px_ph @ dh_dtd

        dx_dt0 = px_px @ dx_dt0 + \
                 px_pt @ dt_dt0 + \
                 px_ph @ dh_dt0

        dx_dx0 = px_px @ dx_dx0

        dt_dtd = dt_dt @ dt_dtd + \
                 dt_dh @ dh_dtd

        dt_dt0 = pt_pt @ dt_dt0

        x = x + h * f
        t = t + h

    return x, t, dx_dx0, dt_dtd, dx_dtd, dt_dt0, dx_dt0


if __name__ == '__main__':
    N = 20
    x0 = 0.5 * np.ones([1], dtype=complex)

    # Perturb x0 for complex-step
    xf, tf, dxf_dx0, dtf_dtd, dxf_dtd , dtf_dt0, dx_dt0 = euler(x0 + 1.e-50j, 0, 2, N)

    print('xf', xf)
    print()
    print('dxf_dx0 = ', dxf_dx0.real)
    print('dxf_dx0 (cs) = ', xf.imag / 1.0E-50)
    print()

    # Perturb t0 for complex-step
    xf, tf, dxf_dx0, dtf_dtd, dxf_dtd , dtf_dt0, dx_dt0 = euler(x0, 0 + 1e-50j, 2, N)

    print('dtf_dt0 = ', dtf_dt0.real)
    print('dtf_dt0 (cs) = ', tf.imag / 1.0E-50)
    print()
    print('dxf_dt0 = ', dx_dt0.real)
    print('dxf_dt0 (cs) = ', xf.imag / 1.0E-50)

    # Perturb tf for complex-step
    xf, tf, dx_dx0, dtf_dtd, dxf_dtd, dtf_dt0, dx_dt0 = euler(x0, 0, 2 + 1e-50j, N)
    print()
    print('dtf_dtd = ', dtf_dtd.real)
    print('dtf_dtd (cs) = ', tf.imag / 1.0E-50)
    print()
    print('dxf_dtd = ', dxf_dtd.real)
    print('dxf_dtd (cs) = ', xf.imag / 1.0E-50)

    # td = 10
    # x0 = 1.0 * np.ones([1], dtype=complex)
    #
    # xf, tf, x_bar, dtf_dtd = euler_rev(x0 + 1e-50j, td, N=3)
    # print('x_bar', x_bar)
    # print('cs = ', xf.imag / 1.0E-50)
