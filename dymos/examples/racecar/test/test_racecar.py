import unittest

try:
    import matplotlib
except ImportError:
    matplotlib = None

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal


@use_tempdirs
class TestRaceCarForDocs(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_racecar_for_docs(self):
        import openmdao.api as om
        import dymos as dm

        from dymos.examples.racecar.combinedODE import CombinedODE
        from dymos.examples.racecar.spline import get_spline, get_track_points
        from dymos.examples.racecar.tracks import ovaltrack  # track curvature imports

        # change track here and in curvature.py. Tracks are defined in tracks.py
        track = ovaltrack

        # generate nodes along the centerline for curvature calculation (different
        # than collocation nodes)
        points = get_track_points(track)

        # fit the centerline spline.
        finespline, gates, gatesd, curv, slope = get_spline(points, s=0.0)
        # by default 10000 points
        s_final = track.get_total_length()

        txs = {'radau': dm.Radau(num_segments=50, order=3, compressed=True),
               'gauss-lobatto': dm.GaussLobatto(num_segments=50, order=3, compressed=True)}

        for tx_name, tx in txs.items():
            with self.subTest(tx_name):
                # Define the OpenMDAO problem
                p = om.Problem(model=om.Group())

                # Define a Trajectory object
                traj = dm.Trajectory()
                p.model.add_subsystem('traj', subsys=traj)

                phase = dm.Phase(ode_class=CombinedODE,
                                 transcription=tx)

                traj.add_phase(name='phase0', phase=phase)

                # Set the time options, in this problem we perform a change of variables. So 'time' is
                # actually 's' (distance along the track centerline)
                # This is done to fix the collocation nodes in space, which saves us the calculation of
                # the rate of change of curvature.
                # The state equations are written with respect to time, the variable change occurs in
                # timeODE.py
                phase.set_integ_var_options(fix_initial=True, fix_duration=True, duration_val=s_final,
                                            name='s', targets=['curv.s'], units='m', duration_ref=s_final,
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
                                  fix_final=False, ref=0.04, rate_continuity=True)  # steering angle
                phase.add_control(name='thrust', units=None, fix_initial=False, fix_final=False, rate_continuity=True)
                # the thrust controls the longitudinal force of the rear tires and
                # is positive while accelerating, negative while braking

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
                phase.add_timeseries_output('t', output_name='time')

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
                p.driver.opt_settings['print_level'] = 0
                p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
                p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
                p.driver.opt_settings['mu_strategy'] = 'monotone'
                p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
                # p.driver.options['print_results'] = False

                # Allow OpenMDAO to automatically determine our sparsity pattern.
                # Doing so can significant speed up the execution of Dymos.
                p.driver.declare_coloring()

                # Setup the problem
                p.setup(check=True)  # force_alloc_complex=True
                # Now that the OpenMDAO problem is setup, we can set the values of the states.

                # Test that set_integ_var_val works.
                # This isn't necessary because we already set the fixed endpoints
                # in set_integ_var_options, but here it's serving as a test of the method.
                phase.set_integ_var_val(initial=0, duration=s_final, units='m')

                # States
                # Nonzero velocity to avoid division by zero errors
                phase.set_state_val('V', 20.0, units='m/s')
                # All other states start at 0
                phase.set_state_val('lambda', 0.0, units='rad')
                phase.set_state_val('omega', 0.0, units='rad/s')
                phase.set_state_val('alpha', 0.0, units='rad')
                phase.set_state_val('ax', 0.0, units='m/s**2')
                phase.set_state_val('ay', 0.0, units='m/s**2')
                phase.set_state_val('n', 0.0, units='m')
                # initial guess for what the final time should be
                phase.set_state_val('t', [0.0, 100.0], units='s')

                # Controls
                # a small amount of thrust can speed up convergence
                phase.set_control_val('delta', 0.0, units='rad')
                phase.set_control_val('thrust', 0.1, units=None)

                dm.run_problem(p, run_driver=True)

                assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1, ...], 22.2657, tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
