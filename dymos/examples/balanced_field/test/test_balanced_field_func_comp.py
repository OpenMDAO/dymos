from distutils.version import LooseVersion
import unittest

import openmdao.api as om
import openmdao.func_api as omf
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao import __version__ as om_version
import dymos as dm
import numpy as np

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


def wrap_ode_func(num_nodes, flight_mode, grad_method='jax', jax_jit=True):
    """
    Returns the metadata from omf needed to create a new ExplciitFuncComp.
    """
    nn = num_nodes
    ode_func = runway_ode if flight_mode == 'runway' else climb_ode

    meta = (omf.wrap(ode_func)
            .add_input('rho', val=1.225, desc='atmospheric density at runway', units='kg/m**3')
            .add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
            .add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)
            .add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
            .add_input('CL_max', val=2.0, desc='maximum lift coefficient for linear fit', units=None)
            .add_input('alpha_max', val=np.radians(10), desc='angle of attack at CL_max', units='rad')
            .add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
            .add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
            .add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
            .add_input('span', val=35.7, desc='Wingspan', units='m')
            .add_input('T', val=1.0, desc='thrust', units='N')

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

    if flight_mode == 'runway':
        meta.add_input('mu_r', val=0.05, desc='runway friction coefficient', units=None)
    else:
        meta.add_input('gam', shape=(nn,), desc='flight path angle', units='rad')
        meta.add_output('gam_dot', shape=(nn,), desc='rate of change of flight path angle',
                        units='rad/s', tags=['dymos.state_rate_source:gam'])
        meta.add_output('h_dot', shape=(nn,), desc='rate of change of altitude', units='m/s',
                        tags=['dymos.state_rate_source:h'])

    meta.declare_coloring('*', method=grad_method)
    meta.declare_partials(of='*', wrt='*', method=grad_method)

    return om.ExplicitFuncComp(meta, use_jax=grad_method == 'jax', use_jit=jax_jit)


@use_tempdirs
class TestBalancedFieldFuncComp(unittest.TestCase):

    @unittest.skipIf(LooseVersion(om_version) < LooseVersion('3.14'), 'requires OpenMDAO >= 3.14')
    @unittest.skipIf(jax is None, 'requires jax and jaxlib')
    @require_pyoptsparse('IPOPT')
    def test_balanced_field_func_comp_radau(self):
        self._run_problem(dm.Radau)

    @unittest.skipIf(LooseVersion(om_version) < LooseVersion('3.14'), 'requires OpenMDAO >= 3.14')
    @unittest.skipIf(jax is None, 'requires jax and jaxlib')
    @require_pyoptsparse('IPOPT')
    def test_balanced_field_func_comp_gl(self):
        self._run_problem(dm.GaussLobatto)

    def _run_problem(self, tx):
        p = om.Problem()

        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        # Use IPOPT if available, with fallback to SLSQP
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.options['print_results'] = True

        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['mu_strategy'] = 'adaptive'

        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['mu_init'] = 0.01
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

        # First Phase: Brake release to V1 - both engines operable
        br_to_v1 = dm.Phase(ode_class=wrap_ode_func, transcription=tx(num_segments=3),
                            ode_init_kwargs={'flight_mode': 'runway'})
        br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0)
        br_to_v1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0)
        br_to_v1.add_state('v', fix_initial=True, lower=0, ref=100.0, defect_ref=100.0)
        br_to_v1.add_parameter('alpha', val=0.0, opt=False, units='deg')
        br_to_v1.add_timeseries_output('*')

        # Second Phase: Rejected takeoff at V1 - no engines operable
        rto = dm.Phase(ode_class=wrap_ode_func, transcription=tx(num_segments=3),
                       ode_init_kwargs={'flight_mode': 'runway'})
        rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        rto.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rto.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        rto.add_parameter('alpha', val=0.0, opt=False, units='deg')
        rto.add_timeseries_output('*')

        # Third Phase: V1 to Vr - single engine operable
        v1_to_vr = dm.Phase(ode_class=wrap_ode_func, transcription=tx(num_segments=3),
                            ode_init_kwargs={'flight_mode': 'runway'})
        v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        v1_to_vr.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        v1_to_vr.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        v1_to_vr.add_parameter('alpha', val=0.0, opt=False, units='deg')
        v1_to_vr.add_timeseries_output('*')

        # Fourth Phase: Rotate - single engine operable
        rotate = dm.Phase(ode_class=wrap_ode_func, transcription=tx(num_segments=3),
                          ode_init_kwargs={'flight_mode': 'runway'})
        rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0)
        rotate.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rotate.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        rotate.add_polynomial_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10, val=[0, 10])
        rotate.add_timeseries_output('*')

        # Fifth Phase: Climb to target speed and altitude at end of runway.
        climb = dm.Phase(ode_class=wrap_ode_func, transcription=tx(num_segments=5),
                         ode_init_kwargs={'flight_mode': 'climb'})
        climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0)
        climb.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        climb.add_state('h', fix_initial=True, lower=0, ref=1.0, defect_ref=1.0)
        climb.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        climb.add_state('gam', fix_initial=True, lower=0, ref=0.05, defect_ref=0.05)
        climb.add_control('alpha', opt=True, units='deg', lower=-10, upper=15, ref=10)
        climb.add_timeseries_output('*')

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_v1', br_to_v1)
        traj.add_phase('rto', rto)
        traj.add_phase('v1_to_vr', v1_to_vr)
        traj.add_phase('rotate', rotate)
        traj.add_phase('climb', climb)

        all_phases = ['br_to_v1', 'v1_to_vr', 'rto', 'rotate', 'climb']
        groundroll_phases = ['br_to_v1', 'v1_to_vr', 'rto', 'rotate']

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={phase: ['m'] for phase in all_phases})

        # Handle parameters which change from phase to phase.
        traj.add_parameter('T_nominal', val=27000 * 2, opt=False, units='lbf', static_target=True,
                           desc='nominal aircraft thrust',
                           targets={'br_to_v1': ['T']})

        traj.add_parameter('T_engine_out', val=27000, opt=False, units='lbf', static_target=True,
                           desc='thrust under a single engine',
                           targets={'v1_to_vr': ['T'], 'rotate': ['T'], 'climb': ['T']})

        traj.add_parameter('T_shutdown', val=0.0, opt=False, units='lbf', static_target=True,
                           desc='thrust when engines are shut down for rejected takeoff',
                           targets={'rto': ['T']})

        traj.add_parameter('mu_r_nominal', val=0.03, opt=False, units=None, static_target=True,
                           desc='nominal runway friction coefficient',
                           targets={'br_to_v1': ['mu_r'], 'v1_to_vr': ['mu_r'],  'rotate': ['mu_r']})

        traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, static_target=True,
                           desc='runway friction coefficient under braking',
                           targets={'rto': ['mu_r']})

        traj.add_parameter('h_runway', val=0., opt=False, units='ft',
                           desc='runway altitude',
                           targets={phase: ['h'] for phase in groundroll_phases})

        # Here we're omitting some constants that are common throughout all phases for the sake of brevity.
        # Their correct defaults are specified in add_input calls to `wrap_ode_func`.

        # Standard "end of first phase to beginning of second phase" linkages
        # Alpha changes from being a parameter in v1_to_vr to a polynomial control
        # in rotate, to a dynamic control in `climb`.
        traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])

        # Less common "final value of r must match at ends of two phases".
        traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
                                    phase_b='climb', var_b='r', loc_b='final',
                                    ref=1000)

        # Define the constraints and objective for the optimal control problem
        v1_to_vr.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=100)

        rto.add_boundary_constraint('v', loc='final', equals=0., ref=100, linear=True)

        rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

        climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
        climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
        climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
        climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        rto.add_objective('r', loc='final', ref=1000.0)

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p.set_val('traj.br_to_v1.t_initial', 0)
        p.set_val('traj.br_to_v1.t_duration', 35)
        p.set_val('traj.br_to_v1.states:r', br_to_v1.interp('r', [0, 2500.0]))
        p.set_val('traj.br_to_v1.states:v', br_to_v1.interp('v', [0, 100.0]))
        p.set_val('traj.br_to_v1.parameters:alpha', 0, units='deg')

        p.set_val('traj.v1_to_vr.t_initial', 35)
        p.set_val('traj.v1_to_vr.t_duration', 35)
        p.set_val('traj.v1_to_vr.states:r', v1_to_vr.interp('r', [2500, 300.0]))
        p.set_val('traj.v1_to_vr.states:v', v1_to_vr.interp('v', [100, 110.0]))
        p.set_val('traj.v1_to_vr.parameters:alpha', 0.0, units='deg')

        p.set_val('traj.rto.t_initial', 35)
        p.set_val('traj.rto.t_duration', 35)
        p.set_val('traj.rto.states:r', rto.interp('r', [2500, 5000.0]))
        p.set_val('traj.rto.states:v', rto.interp('v', [110, 0]))
        p.set_val('traj.rto.parameters:alpha', 0.0, units='deg')

        p.set_val('traj.rotate.t_initial', 70)
        p.set_val('traj.rotate.t_duration', 5)
        p.set_val('traj.rotate.states:r', rotate.interp('r', [1750, 1800.0]))
        p.set_val('traj.rotate.states:v', rotate.interp('v', [80, 85.0]))
        p.set_val('traj.rotate.polynomial_controls:alpha', 0.0, units='deg')

        p.set_val('traj.climb.t_initial', 75)
        p.set_val('traj.climb.t_duration', 15)
        p.set_val('traj.climb.states:r', climb.interp('r', [5000, 5500.0]), units='ft')
        p.set_val('traj.climb.states:v', climb.interp('v', [160, 170.0]), units='kn')
        p.set_val('traj.climb.states:h', climb.interp('h', [0, 35.0]), units='ft')
        p.set_val('traj.climb.states:gam', climb.interp('gam', [0, 5.0]), units='deg')
        p.set_val('traj.climb.controls:alpha', 5.0, units='deg')

        dm.run_problem(p, run_driver=True, simulate=True)

        assert_near_equal(p.get_val('traj.rto.states:r')[-1], 2197.7, tolerance=0.01)
