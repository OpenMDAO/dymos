import unittest

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

import openmdao.api as om
import dymos as dm

from dymos.examples.racecar.combinedODE import CombinedODE
from dymos.examples.racecar.spline import get_spline, get_track_points
from dymos.examples.racecar.tracks import ovaltrack  # track curvature imports


def _run_racecar_problem(transcription, timeseries=False):
    # change track here and in curvature.py. Tracks are defined in tracks.py
    track = ovaltrack

    # generate nodes along the centerline for curvature calculation (different
    # than collocation nodes)
    points = get_track_points(track)

    # fit the centerline spline.
    finespline, gates, gatesd, curv, slope = get_spline(points, s=0.0)
    # by default 10000 points
    s_final = track.get_total_length()

    # Define the OpenMDAO problem
    p = om.Problem(model=om.Group())

    # Define a Trajectory object
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)

    # Define a Dymos Phase object with radau Transcription
    phase = dm.Phase(ode_class=CombinedODE,
                     transcription=transcription(num_segments=50, order=3, compressed=True))

    traj.add_phase(name='phase0', phase=phase)

    # Set the time options, in this problem we perform a change of variables. So 'time' is
    # actually 's' (distance along the track centerline)
    # This is done to fix the collocation nodes in space, which saves us the calculation of
    # the rate of change of curvature.
    # The state equations are written with respect to time, the variable change occurs in
    # timeODE.py
    phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=s_final,
                           targets=['curv.s'], units='m', duration_ref=s_final,
                           duration_ref0=10)

    # Define states
    phase.add_state('t', fix_initial=True, fix_final=False, units='s', lower=0,
                    rate_source='dt_ds', ref=100)  # time
    phase.add_state('n', fix_initial=False, fix_final=False, units='m', upper=4.0, lower=-4.0,
                    rate_source='dn_ds', targets=['n'],
                    ref=4.0)  # normal distance to centerline. The bounds on n define the
    # width of the track
    phase.add_state('V', fix_initial=False, fix_final=False, units='m/s', ref=40, ref0=5,
                    rate_source='dV_ds', targets=['V'])  # velocity
    phase.add_state('alpha', fix_initial=False, fix_final=False, units='rad',
                    rate_source='dalpha_ds', targets=['alpha'],
                    ref=0.15)  # vehicle heading angle with respect to centerline
    phase.add_state('lambda', fix_initial=False, fix_final=False, units='rad',
                    rate_source='dlambda_ds', targets=['lambda'],
                    ref=0.01)  # vehicle slip angle, or angle between the axis of the vehicle
    # and velocity vector (all cars drift a little)
    phase.add_state('omega', fix_initial=False, fix_final=False, units='rad/s',
                    rate_source='domega_ds', targets=['omega'], ref=0.3)  # yaw rate
    phase.add_state('ax', fix_initial=False, fix_final=False, units='m/s**2',
                    rate_source='dax_ds', targets=['ax'], ref=8)  # longitudinal acceleration
    phase.add_state('ay', fix_initial=False, fix_final=False, units='m/s**2',
                    rate_source='day_ds', targets=['ay'], ref=8)  # lateral acceleration

    # Define Controls
    phase.add_control(name='delta', units='rad', lower=None, upper=None, fix_initial=False,
                      fix_final=False, targets=['delta'], ref=0.04)  # steering angle
    phase.add_control(name='thrust', units=None, fix_initial=False, fix_final=False, targets=[
        'thrust'])  # the thrust controls the longitudinal force of the rear tires and is
    # positive while accelerating, negative while braking

    # Performance Constraints
    pmax = 960000  # W
    phase.add_path_constraint('power', upper=pmax, ref=100000)  # engine power limit

    # The following four constraints are the tire friction limits, with 'rr' designating the
    # rear right wheel etc. This limit is computed in tireConstraintODE.py
    phase.add_path_constraint('c_rr', upper=1)
    phase.add_path_constraint('c_rl', upper=1)
    phase.add_path_constraint('c_fr', upper=1)
    phase.add_path_constraint('c_fl', upper=1)

    # Some of the vehicle design parameters are available to set here. Other parameters can
    # be found in their respective ODE files.
    phase.add_parameter('M', val=800.0, units='kg', opt=False,
                        targets=['car.M', 'tire.M', 'tireconstraint.M', 'normal.M'],
                        static_target=True)  # vehicle mass
    phase.add_parameter('beta', val=0.62, units=None, opt=False, targets=['tire.beta'],
                        static_target=True)  # brake bias
    phase.add_parameter('CoP', val=1.6, units='m', opt=False, targets=['normal.CoP'],
                        static_target=True)  # center of pressure location
    phase.add_parameter('h', val=0.3, units='m', opt=False, targets=['normal.h'],
                        static_target=True)  # center of gravity height
    phase.add_parameter('chi', val=0.5, units=None, opt=False, targets=['normal.chi'],
                        static_target=True)  # roll stiffness
    phase.add_parameter('ClA', val=4.0, units='m**2', opt=False, targets=['normal.ClA'],
                        static_target=True)  # downforce coefficient*area
    phase.add_parameter('CdA', val=2.0, units='m**2', opt=False, targets=['car.CdA'],
                        static_target=True)  # drag coefficient*area

    # Minimize final time.
    # note that we use the 'state' time instead of Dymos 'time'
    phase.add_objective('t', loc='final')

    # Add output timeseries
    if timeseries:
        phase.add_timeseries_output('*')

    # Link the states at the start and end of the phase in order to ensure a continous lap
    traj.link_phases(phases=['phase0', 'phase0'],
                     vars=['V', 'n', 'alpha', 'omega', 'lambda', 'ax', 'ay'],
                     locs=('final', 'initial'))

    # Set the driver. IPOPT or SNOPT are recommended but SLSQP might work.
    p.driver = om.pyOptSparseDriver(optimizer='IPOPT')

    p.driver.opt_settings['mu_init'] = 1e-3
    p.driver.opt_settings['max_iter'] = 500
    p.driver.opt_settings['acceptable_tol'] = 1e-3
    p.driver.opt_settings['constr_viol_tol'] = 1e-3
    p.driver.opt_settings['compl_inf_tol'] = 1e-3
    p.driver.opt_settings['acceptable_iter'] = 0
    p.driver.opt_settings['tol'] = 1e-3
    p.driver.opt_settings['nlp_scaling_method'] = 'none'
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence

    # Allow OpenMDAO to automatically determine our sparsity pattern.
    # Doing so can significant speed up the execution of Dymos.
    p.driver.declare_coloring()

    # Setup the problem
    p.setup(check=True)  # force_alloc_complex=True
    # Now that the OpenMDAO problem is setup, we can set the values of the states.

    # States
    # non-zero velocity in order to protect against 1/0 errors.
    p.set_val('traj.phase0.states:V', phase.interp('V', [20, 20]), units='m/s')
    p.set_val('traj.phase0.states:lambda', phase.interp('lambda', [0.0, 0.0]), units='rad')
    # all other states start at 0
    p.set_val('traj.phase0.states:omega', phase.interp('omega', [0.0, 0.0]), units='rad/s')
    p.set_val('traj.phase0.states:alpha', phase.interp('alpha', [0.0, 0.0]), units='rad')
    p.set_val('traj.phase0.states:ax', phase.interp('ax', [0.0, 0.0]), units='m/s**2')
    p.set_val('traj.phase0.states:ay', phase.interp('ay', [0.0, 0.0]), units='m/s**2')
    p.set_val('traj.phase0.states:n', phase.interp('n', [0.0, 0.0]), units='m')
    # initial guess for what the final time should be
    p.set_val('traj.phase0.states:t', phase.interp('t', [0.0, 100.0]), units='s')

    # Controls
    p.set_val('traj.phase0.controls:delta', phase.interp('delta', [0.0, 0.0]), units='rad')
    p.set_val('traj.phase0.controls:thrust', phase.interp('thrust', [0.1, 0.1]), units=None)
    # a small amount of thrust can speed up convergence

    dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
    print('Optimization finished')

    t = p.get_val('traj.phase0.timeseries.states:t')
    assert_near_equal(t[-1], 22.2657, tolerance=0.01)


@use_tempdirs
class BenchmarkRacecar(unittest.TestCase):
    """ Benchmarks for various permutations of the racecar problem."""

    def benchmark_gausslobatto_notimeseries(self):
        _run_racecar_problem(dm.GaussLobatto, timeseries=False)

    def benchmark_gausslobatto_timeseries(self):
        _run_racecar_problem(dm.GaussLobatto, timeseries=True)

    def benchmark_radau_notimeseries(self):
        _run_racecar_problem(dm.Radau, timeseries=False)

    def benchmark_radau_timeseries(self):
        _run_racecar_problem(dm.Radau, timeseries=True)
