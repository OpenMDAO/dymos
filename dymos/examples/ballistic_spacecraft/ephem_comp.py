from functools import partial

import jax
import jax.numpy as jnp
import openmdao.api as om

# Approximate Keplerian elements of Earth and Mars at J2000
_EARTH_ELEMENTS = jnp.array([[1.4959826115044251e+08,
                              1.6711230000000001e-02,
                              -2.6720990848033185e-07,
                              1.7966014740491711e+00,
                              0.0000000000000000e+00,
                              -4.3163916976385941e-02]])


_MARS_ELEMENTS = jnp.array([[2.2794382242757303e+08,
                             9.3394099999999994e-02,
                             3.2283205424889293e-02,
                             -1.2826977680479821e+00,
                             8.6497712974974172e-01,
                             3.3806682838789720e-01]])

_ELEMENTS = jnp.vstack([_EARTH_ELEMENTS, _MARS_ELEMENTS])

KMPAU = 149597870.700  # Kilometers per Astronomical Unit, AU 2012 Resolution B1
MU_SUN = 1.32712440041279419E11  # Gravitational parameter of sun (km**3/s**2)


def _solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 20) -> float:
    """
    Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E
    using Newton-Raphson iteration with jax.lax.scan for AD compatibility.
    """
    # Initial guess: use jax.lax.cond for JIT compatibility
    E = M - e * jnp.sin(M)

    def body_fn(E, _):
        # Newton-Raphson iteration
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        E_new = E - f / fp
        return E_new, E_new

    # Run fixed number of iterations using scan (compatible with reverse-mode AD)
    E_final, _ = jax.lax.scan(body_fn, E, None, length=max_iter)
    return E_final


_solve_kepler_vec = jax.vmap(_solve_kepler, in_axes=(0, 0, None, None))


def ephemeris(elements: jnp.ndarray, t: jnp.ndarray, mu=MU_SUN) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert orbital elements to Cartesian position and velocity vectors at time t.

    Parameters
    ----------
    elements : jnp.ndarray
        A 2D array of the 6 orbital elements of each body:
        - semi-major axis (distance units)
        - eccentricity (unitless)
        - inclination (radians)
        - right ascension of ascending node (radians)
        - argument of periapsis (radians)
        - mean anomaly at the year 0 epoch (radians)
    time : float
        The time at which the cartesian state is requested.
    mu : float
        The gravitational parameter of the central body.

    Returns
    -------
    r : jnp.ndarray
        Cartesian position in distance units.
    v : jnp.ndarray
        Cartesian velocity in distance units / time units.

    """
    a, e, i, raan, argp, M0 = elements[:, 0], elements[:, 1], elements[:, 2], elements[:, 3], elements[:, 4], elements[:, 5]
    mu = MU_SUN

    # Mean motion
    n = jnp.sqrt(mu / a**3)

    # Mean anomaly at time t
    M = M0 + n * t

    # Solve for eccentric anomaly
    E = _solve_kepler_vec(M, e, 1.0E-10, 15)

    # True anomaly
    theta = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + e) * jnp.sin(E / 2.0),
        jnp.sqrt(1.0 - e) * jnp.cos(E / 2.0)
    )

    # Distance
    r_mag = a * (1.0 - e**2) / (1.0 + e * jnp.cos(theta))

    # Velocity magnitude
    v_mag = jnp.sqrt(2.0 * mu / r_mag - mu / a)

    # Flight path angle
    gamma = jnp.arctan2(e * jnp.sin(theta), 1.0 + e * jnp.cos(theta))

    # Position in orbital plane
    cos_theta_argp = jnp.cos(theta + argp)
    sin_theta_argp = jnp.sin(theta + argp)
    cos_raan = jnp.cos(raan)
    sin_raan = jnp.sin(raan)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)

    x = r_mag * (cos_theta_argp * cos_raan - sin_theta_argp * cos_i * sin_raan)
    y = r_mag * (cos_theta_argp * sin_raan + sin_theta_argp * cos_i * cos_raan)
    z = r_mag * sin_theta_argp * sin_i

    # Velocity in orbital plane
    cos_theta_omega_gamma = jnp.cos(theta + argp - gamma)
    sin_theta_omega_gamma = jnp.sin(theta + argp - gamma)

    vx = v_mag * (-sin_theta_omega_gamma * cos_raan - cos_theta_omega_gamma * cos_i * sin_raan)
    vy = v_mag * (-sin_theta_omega_gamma * sin_raan + cos_theta_omega_gamma * cos_i * cos_raan)
    vz = v_mag * cos_theta_omega_gamma * sin_i

    # Stack to (n, 3) shape: each row is [x, y, z] for one body
    r = jnp.vstack([jnp.atleast_2d(x), jnp.atleast_2d(y), jnp.atleast_2d(z)]).T
    v = jnp.vstack([jnp.atleast_2d(vx), jnp.atleast_2d(vy), jnp.atleast_2d(vz)]).T

    return r, v


# Create a partial function with elements and mu fixed
# Then vmap over time (second argument after partial, which becomes first argument after the partial.)
_ephem_func = jax.vmap(partial(ephemeris, _ELEMENTS, mu=MU_SUN), in_axes=(0,))


class EphemerisComp(om.JaxExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_bodies', types=int, default=2)

    def setup(self):
        nn = self.options['num_nodes']
        nb = self.options['num_bodies']
        self.add_input('t', shape=(nn,), units='s')
        # Output shape: (num_times, num_bodies, 3)
        self.add_output('r', shape=(nn, nb, 3), units='km')
        self.add_output('v', shape=(nn, nb, 3), units='km/s')

    def compute_primal(self, t):
        r, v = _ephem_func(t)
        # r and v should have shape (num_times, num_bodies, 3)
        return r, v


if __name__ == '__main__':

    with jax.disable_jit(True):
        p = om.Problem()

        N = 50

        p.model.add_subsystem('ephem', EphemerisComp(num_nodes=N), promotes=['*'])

        p.setup()

        p.set_val('t', jnp.linspace(0, 360 * 86400, N))

        p.run_model()

        p.model.list_vars(print_arrays=True)

        r_earth = p.get_val('r')[:, 0, :]
        r_mars = p.get_val('r')[:, 1, :]

        import matplotlib.pyplot as plt

        plt.plot(r_earth[:, 0], r_earth[:, 1])
        plt.plot(r_mars[:, 0], r_mars[:, 1])

        plt.show()
