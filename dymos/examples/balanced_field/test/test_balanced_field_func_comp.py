import unittest

import openmdao.api as om
import openmdao.func_api as omf
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
import dymos as dm
import numpy as np

from dymos.examples.balanced_field.balanced_field_length import make_balanced_field_length_problem

try:
    import jax
except ImportError:
    jax = None


def runway_ode(rho, S, CD0, CL0, CL_max, alpha_max, h_w, AR, e, span, T, mu_r, m, v, h, alpha):
    g = 9.80665
    W = m * g
    v_stall = np.sqrt(2 * W / rho / S / CL_max)
    v_over_v_stall = v / v_stall

    CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
    K_nom = 1.0 / (np.pi * AR * e)
    b = span / 2.0
    fact = ((h + h_w) / b) ** 1.5
    K = K_nom * 33 * fact / (1.0 + 33 * fact)

    q = 0.5 * rho * v ** 2
    L = q * S * CL
    D = q * S * (CD0 + K * CL ** 2)

    # Compute the downward force on the landing gear
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)

    # Runway normal force
    F_r = m * g - L * calpha - T * salpha

    # Compute the dynamics
    v_dot = (T * calpha - D - F_r * mu_r) / m
    r_dot = v

    return CL, q, L, D, K, F_r, v_dot, r_dot, W, v_stall, v_over_v_stall


def climb_ode(rho, S, CD0, CL0, CL_max, alpha_max, h_w, AR, e, span, T, m, v, h, alpha, gam):
    g = 9.80665
    W = m * g
    v_stall = np.sqrt(2 * W / rho / S / CL_max)
    v_over_v_stall = v / v_stall

    CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
    K_nom = 1.0 / (np.pi * AR * e)
    b = span / 2.0
    fact = ((h + h_w) / b) ** 1.5
    K = K_nom * 33 * fact / (1.0 + 33 * fact)

    q = 0.5 * rho * v ** 2
    L = q * S * CL
    D = q * S * (CD0 + K * CL ** 2)

    # Compute the downward force on the landing gear
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)

    # Runway normal force
    F_r = m * g - L * calpha - T * salpha

    # Compute the dynamics
    cgam = np.cos(gam)
    sgam = np.sin(gam)
    v_dot = (T * calpha - D) / m - g * sgam
    gam_dot = (T * salpha + L) / (m * v) - (g / v) * cgam
    h_dot = v * sgam
    r_dot = v * cgam

    return CL, q, L, D, K, F_r, v_dot, r_dot, W, v_stall, v_over_v_stall, gam_dot, h_dot


def wrap_ode_func(num_nodes, mode, grad_method='jax', jax_jit=True):
    """
    Returns the metadata from omf needed to create a new ExplciitFuncComp.
    """
    nn = num_nodes
    ode_func = runway_ode if mode == 'runway' else climb_ode

    meta = (omf.wrap(ode_func)
            .defaults(shape=(1,))
            .add_input('rho', val=1.225, desc='atmospheric density at runway', units='kg/m**3', shape=(1, ))
            .add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2', shape=(1, ))
            .add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None, shape=(1, ))
            .add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None, shape=(1, ))
            .add_input('CL_max', val=2.0, desc='maximum lift coefficient for linear fit', units=None, shape=(1, ))
            .add_input('alpha_max', val=np.radians(10), desc='angle of attack at CL_max', units='rad', shape=(1, ))
            .add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m', shape=(1, ))
            .add_input('AR', val=9.45, desc='wing aspect ratio', units=None, shape=(1, ))
            .add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None, shape=(1, ))
            .add_input('span', val=35.7, desc='Wingspan', units='m', shape=(1, ))
            .add_input('T', val=1.0, desc='thrust', units='N', shape=(1, ))

            # Dynamic inputs (can assume a different value at every node)
            .add_input('m', shape=(nn,), desc='aircraft mass', units='kg')
            .add_input('v', shape=(nn,), desc='aircraft true airspeed', units='m/s')
            .add_input('h', shape=(nn,), desc='altitude', units='m')
            .add_input('alpha', shape=(nn,), desc='angle of attack', units='rad')

            # Outputs
            .add_output('CL', shape=(nn,), desc='lift coefficient', units=None)
            .add_output('q', shape=(nn,), desc='dynamic pressure', units='Pa')
            .add_output('L', shape=(nn,), desc='lift force', units='N')
            .add_output('D', shape=(nn,), desc='drag force', units='N')
            .add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)
            .add_output('F_r', shape=(nn,), desc='runway normal force', units='N')
            .add_output('v_dot', shape=(nn,), desc='rate of change of speed', units='m/s**2',
                        tags=['dymos.state_rate_source:v'])
            .add_output('r_dot', shape=(nn,), desc='rate of change of range', units='m/s',
                        tags=['dymos.state_rate_source:r'])
            .add_output('W', shape=(nn,), desc='aircraft weight', units='N')
            .add_output('v_stall', shape=(nn,), desc='stall speed', units='m/s')
            .add_output('v_over_v_stall', shape=(nn,), desc='stall speed ratio', units=None)
            )

    if mode == 'runway':
        meta.add_input('mu_r', val=0.05, desc='runway friction coefficient', units=None, shape=(1,))
    else:
        meta.add_input('gam', shape=(nn,), desc='flight path angle', units='rad')
        meta.add_output('gam_dot', shape=(nn,), desc='rate of change of flight path angle',
                        units='rad/s', tags=['dymos.state_rate_source:gam'])
        meta.add_output('h_dot', shape=(nn,), desc='rate of change of altitude', units='m/s',
                        tags=['dymos.state_rate_source:h'])

    meta.declare_coloring('*', method=grad_method)
    meta.declare_partials(of='*', wrt='*', method=grad_method)

    return om.ExplicitFuncComp(meta, derivs_method=grad_method, use_jit=jax_jit)


@use_tempdirs
class TestBalancedFieldFuncComp(unittest.TestCase):

    @unittest.skipIf(jax is None, 'requires jax and jaxlib')
    @require_pyoptsparse('IPOPT')
    def test_balanced_field_func_comp_radau(self):
        p = make_balanced_field_length_problem(ode_class=wrap_ode_func,
                                               tx=dm.Radau(num_segments=3))

        dm.run_problem(p, run_driver=True, simulate=True)

        assert_near_equal(p.get_val('traj.rto.states:r')[-1], 2197.7, tolerance=0.01)

    @unittest.skipIf(jax is None, 'requires jax and jaxlib')
    @require_pyoptsparse('IPOPT')
    def test_balanced_field_func_comp_gl(self):
        p = make_balanced_field_length_problem(ode_class=wrap_ode_func,
                                               tx=dm.GaussLobatto(num_segments=3))

        dm.run_problem(p, run_driver=True, simulate=True)

        assert_near_equal(p.get_val('traj.rto.states:r')[-1], 2197.7, tolerance=0.01)
