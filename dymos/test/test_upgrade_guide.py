import unittest

import openmdao.api as om

import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
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

    def test_parameter_no_timeseries(self):
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

    def test_simplified_ode_timeseries_output(self):
        """
        # upgrade_doc: begin simplified_ode_output_timeseries
        phase.add_timeseries_output('tas_comp.TAS', shape=(1,), units='m/s')
        # upgrade_doc: end simplified_ode_output_timeseries
        """
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=1,
                                        order=13,
                                        compressed=False)
        phase = dm.Phase(ode_class=AircraftODE, transcription=transcription)
        p.model.add_subsystem('phase0', phase)

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', om.IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        phase.set_time_options(initial_bounds=(0, 0),
                               duration_bounds=(3600, 3600),
                               duration_ref=3600)

        phase.set_time_options(initial_bounds=(0, 0),
                               duration_bounds=(3600, 3600),
                               duration_ref=3600)

        phase.add_state('range', units='km', fix_initial=True, fix_final=False, scaler=0.01,
                        rate_source='range_rate_comp.dXdt:range',
                        defect_scaler=0.01)
        phase.add_state('mass_fuel', units='kg', fix_final=True, upper=20000.0, lower=0.0,
                        rate_source='propulsion.dXdt:mass_fuel',
                        scaler=1.0E-4, defect_scaler=1.0E-2)
        phase.add_state('alt',
                        rate_source='climb_rate',
                        units='km', fix_initial=True)

        phase.add_control('mach',  targets=['tas_comp.mach', 'aero.mach'], units=None, opt=False)

        phase.add_control('climb_rate', targets=['gam_comp.climb_rate'], units='m/s', opt=False)

        phase.add_parameter('S',
                            targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'],
                            units='m**2')

        phase.add_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
        phase.add_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0, shape=(1,))

        # upgrade_doc: begin simplified_ode_output_timeseries
        phase.add_timeseries_output('tas_comp.TAS')
        # upgrade_doc: end simplified_ode_output_timeseries

        p.model.connect('assumptions.S', 'phase0.parameters:S')
        p.model.connect('assumptions.mass_empty', 'phase0.parameters:mass_empty')
        p.model.connect('assumptions.mass_payload', 'phase0.parameters:mass_payload')

        phase.add_objective('time', loc='final', ref=3600)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.515132 * 3600.0
        p['phase0.states:range'] = phase.interpolate(ys=(0, 1296.4), nodes='state_input')
        p['phase0.states:mass_fuel'] = phase.interpolate(ys=(12236.594555, 0), nodes='state_input')
        p['phase0.states:alt'] = 5.0
        p['phase0.controls:mach'] = 0.8
        p['phase0.controls:climb_rate'] = 0.0

        p['assumptions.S'] = 427.8
        p['assumptions.mass_empty'] = 0.15E6
        p['assumptions.mass_payload'] = 84.02869 * 400

        dm.run_problem(p)

        time = p.get_val('phase0.timeseries.time')
        tas = p.get_val('phase0.timeseries.TAS', units='km/s')
        range = p.get_val('phase0.timeseries.states:range')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)

        exp_out = phase.simulate()

        time = exp_out.get_val('phase0.timeseries.time')
        tas = exp_out.get_val('phase0.timeseries.TAS', units='km/s')
        range = exp_out.get_val('phase0.timeseries.states:range')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)

    def test_glob_timeseries_outputs(self):
        """
        # upgrade_doc: begin glob_timeseries_outputs
        phase.add_timeseries_output('aero.mach', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CD0', shape=(1,), units=None)
        phase.add_timeseries_output('aero.kappa', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CLa', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CL', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CD', shape=(1,), units=None)
        phase.add_timeseries_output('aero.q', shape=(1,), units='N/m**2')
        phase.add_timeseries_output('aero.f_lift', shape=(1,), units='N')
        phase.add_timeseries_output('aero.f_drag', shape=(1,), units='N')
        # upgrade_doc: end glob_timeseries_outputs
        """
        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring(tol=1.0E-12)

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                         transcription=dm.GaussLobatto(num_segments=15, compressed=True))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                        ref=1.0E3, defect_ref=1.0E3, units='m',
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m',
                        rate_source='flight_dynamics.h_dot')

        phase.add_state('v', fix_initial=True, lower=10.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m/s',
                        rate_source='flight_dynamics.v_dot')

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5,
                        ref=1.0, defect_ref=1.0, units='rad',
                        rate_source='flight_dynamics.gam_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                        ref=1.0E3, defect_ref=1.0E3, units='kg',
                        rate_source='prop.m_dot')

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False, targets=['alpha'])

        phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
        phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, shape=(1,))
        phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

        phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8, shape=(1,))

        phase.add_objective('time', loc='final', ref=1.0)

        p.model.linear_solver = om.DirectSolver()

        # upgrade_doc: begin glob_timeseries_outputs
        phase.add_timeseries_output('aero.*')
        # upgrade_doc: end glob_timeseries_outputs

        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 500

        p['traj.phase0.states:r'] = phase.interpolate(ys=[0.0, 50000.0], nodes='state_input')
        p['traj.phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
        p['traj.phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
        p['traj.phase0.states:m'] = phase.interpolate(ys=[19030.468, 10000.], nodes='state_input')
        p['traj.phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

        #
        # Solve for the optimal trajectory
        #
        p.run_model()

        outputs = p.model.list_outputs(units=True, out_stream=None, prom_name=True)
        op_dict = {options['prom_name']: options['units'] for abs_name, options in outputs}

        for name, units in [('mach', None), ('CD0', None), ('kappa', None), ('CLa', None),
                            ('CL', None), ('CD', None), ('q', 'N/m**2'), ('f_lift', 'N'),
                            ('f_drag', 'N')]:
            self.assertEqual(op_dict[f'traj.phase0.timeseries.{name}'], units)

    def test_sequence_timeseries_outputs(self):
        """
        # upgrade_doc: begin sequence_timeseries_outputs
        phase.add_timeseries_output('aero.mach', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CD0', shape=(1,), units=None)
        phase.add_timeseries_output('aero.kappa', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CLa', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CL', shape=(1,), units=None)
        phase.add_timeseries_output('aero.CD', shape=(1,), units=None)
        phase.add_timeseries_output('aero.q', shape=(1,), units='N/m**2')
        phase.add_timeseries_output('aero.f_lift', shape=(1,), units='lbf')
        phase.add_timeseries_output('aero.f_drag', shape=(1,), units='N')
        phase.add_timeseries_output('prop.thrust', shape=(1,), units='lbf')
        # upgrade_doc: end sequence_timeseries_outputs
        # upgrade_doc: begin state_endpoint_values
        final_range = p.get_val('traj.phase0.final_conditions.states:x0++')
        final_alpha = p.get_val('traj.phase0.final_conditions.controls:alpha++')
        # upgrade_doc: end state_endpoint_values
        """
        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring(tol=1.0E-12)

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                         transcription=dm.GaussLobatto(num_segments=15, compressed=True))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                        ref=1.0E3, defect_ref=1.0E3, units='m',
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m',
                        rate_source='flight_dynamics.h_dot')

        phase.add_state('v', fix_initial=True, lower=10.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m/s',
                        rate_source='flight_dynamics.v_dot')

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5,
                        ref=1.0, defect_ref=1.0, units='rad',
                        rate_source='flight_dynamics.gam_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                        ref=1.0E3, defect_ref=1.0E3, units='kg',
                        rate_source='prop.m_dot')

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False, targets=['alpha'])

        phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
        phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, shape=(1,))
        phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

        phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8, shape=(1,))

        phase.add_objective('time', loc='final', ref=1.0)

        p.model.linear_solver = om.DirectSolver()

        # upgrade_doc: begin sequence_timeseries_outputs
        phase.add_timeseries_output(['aero.*', 'prop.thrust'],
                                    units={'aero.f_lift': 'lbf', 'prop.thrust': 'lbf'})
        # upgrade_doc: end sequence_timeseries_outputs

        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 500

        p['traj.phase0.states:r'] = phase.interpolate(ys=[0.0, 50000.0], nodes='state_input')
        p['traj.phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
        p['traj.phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
        p['traj.phase0.states:m'] = phase.interpolate(ys=[19030.468, 10000.], nodes='state_input')
        p['traj.phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

        p.run_model()

        # upgrade_doc: begin state_endpoint_values
        final_range = p.get_val('traj.phase0.timeseries.states:r')[-1, ...]
        final_alpha = p.get_val('traj.phase0.timeseries.controls:alpha')[-1, ...]
        # upgrade_doc: end state_endpoint_values
        self.assertEqual(final_range, 50000.0)
        self.assertEqual(final_alpha, 0.0)

        outputs = p.model.list_outputs(units=True, out_stream=None, prom_name=True)
        op_dict = {options['prom_name']: options['units'] for abs_name, options in outputs}

        for name, units in [('mach', None), ('CD0', None), ('kappa', None), ('CLa', None),
                            ('CL', None), ('CD', None), ('q', 'N/m**2'), ('f_lift', 'lbf'),
                            ('f_drag', 'N'), ('thrust', 'lbf')]:
            self.assertEqual(op_dict[f'traj.phase0.timeseries.{name}'], units)
