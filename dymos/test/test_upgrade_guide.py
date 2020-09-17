import unittest



class TestUpgrade_0_16_0(unittest.TestCase):

    def test_parameters(self):
        """
        # upgrade_doc: begin set_val
        p.set_val('traj.phase0.design_parameters:thrust', 2.1, units='MN')
        # upgrade_doc: end set_val
        # upgrade_doc: begin parameter_timeseries
        thrust = p.get_val('traj.phase0.timeseries.design_parameters:thrust')
        # upgrade_doc: end parameter_timeseries
        """
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        #
        # Setup and solve the optimal control problem
        #
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE

        #
        # Initialize our Trajectory and Phase
        #
        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=LaunchVehicleODE,
                         ode_init_kwargs={'central_body': 'earth'},
                         transcription=dm.GaussLobatto(num_segments=12, order=3, compressed=False))

        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        #
        # Set the options for the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 500))

        phase.add_state('x', fix_initial=True, ref=1.0E5, defect_ref=1.0,
                        rate_source='eom.xdot', units='m')
        phase.add_state('y', fix_initial=True, ref=1.0E5, defect_ref=1.0,
                        rate_source='eom.ydot', targets=['atmos.y'], units='m')
        phase.add_state('vx', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                        rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
        phase.add_state('vy', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                        rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
        phase.add_state('m', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                        rate_source='eom.mdot', targets=['eom.m'], units='kg')

        phase.add_control('theta', units='rad', lower=-1.57, upper=1.57, targets=['eom.theta'])
        phase.add_parameter('thrust', units='N', opt=False, val=2100000.0, targets=['eom.thrust'])

        #
        # Set the options for our constraints and objective
        #
        phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
        phase.add_boundary_constraint('vx', loc='final', equals=7796.6961)
        phase.add_boundary_constraint('vy', loc='final', equals=0)

        phase.add_objective('time', loc='final', scaler=0.01)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup and set initial values
        #
        p.setup(check=True)

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 150.0)
        p.set_val('traj.phase0.states:x', phase.interpolate(ys=[0, 1.15E5], nodes='state_input'))
        p.set_val('traj.phase0.states:y', phase.interpolate(ys=[0, 1.85E5], nodes='state_input'))
        p.set_val('traj.phase0.states:vx', phase.interpolate(ys=[0, 7796.6961], nodes='state_input'))
        p.set_val('traj.phase0.states:vy', phase.interpolate(ys=[1.0E-6, 0], nodes='state_input'))
        p.set_val('traj.phase0.states:m', phase.interpolate(ys=[117000, 1163], nodes='state_input'))
        p.set_val('traj.phase0.controls:theta', phase.interpolate(ys=[1.5, -0.76], nodes='control_input'))

        # upgrade_doc: begin set_val
        p.set_val('traj.phase0.parameters:thrust', 2.1, units='MN')
        # upgrade_doc: end set_val

        #
        # Solve the Problem
        #
        dm.run_problem(p)

        #
        # Check the results.
        #
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 143, tolerance=0.05)
        assert_near_equal(p.get_val('traj.phase0.timeseries.states:y')[-1], 1.85E5, 1e-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.states:vx')[-1], 7796.6961, 1e-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.states:vy')[-1], 0, 1e-4)

        # upgrade_doc: begin parameter_timeseries
        thrust = p.get_val('traj.phase0.timeseries.parameters:thrust')
        # upgrade_doc: end parameter_timeseries
        nn = phase.options['transcription'].grid_data.num_nodes
        assert_near_equal(thrust, 2.1E6 * np.ones([nn, 1]))

    def test_parameter_no_include_timeseries(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        # upgrade_doc: begin parameter_no_timeseries
        phase.add_parameter('g', opt=False, units='m/s**2', include_timeseries=False)
        # upgrade_doc: end parameter_no_timeseries

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        with self.assertRaises(KeyError):
            p.get_val('phase0.timeseries.parameters:g}')
