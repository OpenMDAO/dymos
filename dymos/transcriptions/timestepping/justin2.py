import numpy as np


class CannonballODE:
    _xdot = np.zeros((4, 1), dtype=complex)
    _dx = np.zeros((4, 4), dtype=complex)
    _dt = np.zeros((4, 1), dtype=complex)

    """
    An ODE that takes the position and velocity as states [x, y, vx, vy] and provides their rates
    """
    @classmethod
    def eval(cls, x, t):
        cls._xdot[0, 0] = x[2]
        cls._xdot[1, 0] = x[3]
        cls._xdot[2, 0] = 0.0
        cls._xdot[3, 0] = -9.80665
        return cls._xdot

    @classmethod
    def dx(cls, x, t):
        """
        The derivative of the rate of x wrt x.
        """
        cls._dx[0, :] = [0, 0, 1, 0]
        cls._dx[1, :] = [0, 0, 0, 1]
        cls._dx[2, :] = [0, 0, 0, 0]
        cls._dx[3, :] = [0, 0, 0, 0]
        return cls._dx

    @classmethod
    def dt(cls, x, t):
        """
        The derivative of the rate of x wrt t.
        """
        return cls._dt

def euler_fwd(ode, x0, t0, td, N):
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
        f = ode.eval(x, t)
        f_x = ode.dx(x, t)
        f_t = ode.dt(x, t)

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

def euler_rev(ode, x0, t0, td, N):
    """
    Euler's method with reverse-computed derivatives.

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

    dx_dx0_bar = np.eye(x_size, dtype=complex).T
    dt_dtd_bar = np.zeros((1, 1), dtype=complex).T
    dx_dtd_bar = np.zeros((x_size, 1), dtype=complex).T
    dt_dt0_bar = np.ones((1, 1), dtype=complex).T
    dx_dt0_bar = np.zeros((x_size, 1), dtype=complex).T

    dh_dt0_bar = np.zeros((1, 1), dtype=complex).T / N
    dh_dtd_bar = np.ones((1, 1), dtype=complex).T / N
    dt_dt_bar = np.ones((1, 1), dtype=complex).T
    dt_dh_bar = np.ones((1, 1), dtype=complex).T

    I_x = np.eye(x_size, dtype=complex)
    I_t = np.eye(1, dtype=complex)

    x_stack = []
    t_stack = []

    x_stack.append(x.copy())
    t_stack.append(t)

    # Compute the final state by propagating forward
    # We save off the state and time from each step so that they
    # can be used to compute the derivatives in a backward pass.
    for i in range(N):
        x += h * ode.eval(x, t)
        x_stack.append(x.copy())
        t += h
        t_stack.append(t)

    for i in range(N):
        x_i = x_stack.pop()
        t_i = t_stack.pop()

        f = ode.eval(x_i, t_i)
        f_x = ode.dx(x_i, t_i)
        f_t = ode.dt(x_i, t_i)

        px_px = I_x + h * f_x
        px_pt = h * f_t
        px_ph = f
        pt_pt = I_t

        # Compute this with the initial values of dx_dx and dt_dtd before they're updated
        dx_dtd_bar = dx_dtd_bar @ px_px.T + \
                     dt_dtd_bar @ px_pt.T + \
                     dh_dtd_bar @ px_ph.T

        dx_dt0_bar = dx_dt0_bar @ px_px.T + \
                     dt_dt0_bar @ px_pt.T + \
                     dh_dt0_bar @ px_ph.T

        dx_dx0_bar = dx_dx0_bar @ px_px.T

        dt_dtd_bar = dt_dtd_bar @ dt_dt_bar.T + \
                     dh_dtd_bar @ dt_dh_bar.T

        dt_dt0_bar = dt_dt_bar @ pt_pt.T

    return x, t, dx_dx0_bar, dt_dtd_bar, dx_dtd_bar, dt_dt0_bar, dx_dt0_bar


if __name__ == '__main__':
    N = 1000
    x0 = np.array([[0, 0, 100, 100]], dtype=complex).T

    # Perturb x0 for complex-step
    x0[2] += 1.0E-50j
    xf, tf, dxf_dx0, dtf_dtd, dxf_dtd, dtf_dt0, dx_dt0 = euler_fwd(CannonballODE, x0, 0, 2 * x0[3,0]/9.80665, N)

    print('xf\n', xf.real)
    print()
    print('dxf_dx0 = \n', dxf_dx0.real)
    print('dxf_dx0 (cs) = \n', xf.imag / 1.0E-50)
    print()

    xf, tf, dxf_dx0, dtf_dtd, dxf_dtd, dtf_dt0, dx_dt0 = euler_rev(CannonballODE, x0, 0, 2 * x0[3,0]/9.80665, N)

    print('xf\n', xf.real)
    print()
    print('dxf_dx0 = \n', dxf_dx0.real)
    print('dxf_dx0 (cs) = \n', xf.imag / 1.0E-50)
    print()

