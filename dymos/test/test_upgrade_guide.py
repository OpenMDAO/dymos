import os
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
        p.driver.declare_coloring(tol=1.0E-12)

        from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE

        #
        # Initialize our Trajectory and Phase
        #
        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=LaunchVehicleODE,
                         transcription=dm.GaussLobatto(num_segments=12, order=3, compressed=False))

        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        #
        # Set the options for the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 500))

        phase.add_state('x', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                        rate_source='xdot')
        phase.add_state('y', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                        rate_source='ydot')
        phase.add_state('vx', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                        rate_source='vxdot')
        phase.add_state('vy', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                        rate_source='vydot')
        phase.add_state('m', fix_initial=True, ref=1.0E3, defect_ref=100.0,
                        rate_source='mdot')

        phase.add_control('theta', units='rad', lower=-1.57, upper=1.57)
        phase.add_parameter('thrust', units='N', opt=False, val=2100000.0)

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

        phase.add_state('x', fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False)

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


@use_tempdirs
class TestUpgrade_0_17_0(unittest.TestCase):

    def test_tags(self):
        """
        # upgrade_doc: begin tag_rate_source
        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')
        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')
        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')
        # upgrade_doc: end tag_rate_source
        # upgrade_doc: begin declare_rate_source
        phase.add_state('x', rate_source='xdot', fix_initial=True, fix_final=True)
        phase.add_state('y', rate_source='ydot', fix_initial=True, fix_final=True)
        phase.add_state('v', rate_source='vdot', fix_initial=True, fix_final=False)
        # upgrade_doc: end declare_rate_source
        """
        import numpy as np
        import openmdao.api as om

        class BrachistochroneODE(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)
                self.options.declare('static_gravity', types=(bool,), default=False,
                                     desc='If True, treat gravity as a static (scalar) input, rather than '
                                          'having different values at each node.')

            def setup(self):
                nn = self.options['num_nodes']
                g_default_val = 9.80665 if self.options['static_gravity'] else 9.80665 * np.ones(nn)

                # Inputs
                self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

                self.add_input('g', val=g_default_val, desc='grav. acceleration', units='m/s/s')

                self.add_input('theta', val=np.ones(nn), desc='angle of wire', units='rad')

                # upgrade_doc: begin tag_rate_source
                self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s',
                                tags=['state_rate_source:x', 'state_units:m'])

                self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s',
                                tags=['state_rate_source:y', 'state_units:m'])

                self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2',
                                tags=['state_rate_source:v', 'state_units:m/s'])
                # upgrade_doc: end tag_rate_source

                self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                                units='m/s')

                # Setup partials
                arange = np.arange(self.options['num_nodes'])
                self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

                self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
                self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

                self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
                self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

                self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
                self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)

                if self.options['static_gravity']:
                    c = np.zeros(self.options['num_nodes'])
                    self.declare_partials(of='vdot', wrt='g', rows=arange, cols=c)
                else:
                    self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)

            def compute(self, inputs, outputs):
                theta = inputs['theta']
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                g = inputs['g']
                v = inputs['v']

                outputs['vdot'] = g * cos_theta
                outputs['xdot'] = v * sin_theta
                outputs['ydot'] = -v * cos_theta
                outputs['check'] = v / sin_theta

            def compute_partials(self, inputs, partials):
                theta = inputs['theta']
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                g = inputs['g']
                v = inputs['v']

                partials['vdot', 'g'] = cos_theta
                partials['vdot', 'theta'] = -g * sin_theta

                partials['xdot', 'v'] = sin_theta
                partials['xdot', 'theta'] = v * cos_theta

                partials['ydot', 'v'] = -cos_theta
                partials['ydot', 'theta'] = v * sin_theta

                partials['check', 'v'] = 1 / sin_theta
                partials['check', 'theta'] = -v * cos_theta / sin_theta ** 2

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.GaussLobatto(num_segments=10)))

        #
        # Set the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        # upgrade_doc: begin declare_rate_source
        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)
        # upgrade_doc: end declare_rate_source

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)
        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj.phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)


@use_tempdirs
class TestUpgrade_0_19_0(unittest.TestCase):

    def tearDown(self):
        if os.path.exists('dymos_solution.db'):
            os.remove('dymos_solution.db')
        if os.path.exists('dymos_simulation.db'):
            os.remove('dymos_simulation.db')

    def _make_problem(self, transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                      compressed=True, optimizer='SLSQP', run_driver=True,
                      force_alloc_complex=False,
                      solve_segments=False):

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring(tol=1.0E-12)

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=num_segments,
                         order=transcription_order,
                         compressed=compressed)

        # upgrade_doc: begin exec_comp_ode
        ode = lambda num_nodes: om.ExecComp(['vdot = g * cos(theta)',
                                             'xdot = v * sin(theta)',
                                             'ydot = -v * cos(theta)'],
                                            g={'value': 9.80665, 'units': 'm/s**2'},
                                            v={'shape': (num_nodes,), 'units': 'm/s'},
                                            theta={'shape': (num_nodes,), 'units': 'rad'},
                                            vdot={'shape': (num_nodes,), 'units': 'm/s**2'},
                                            xdot={'shape': (num_nodes,), 'units': 'm/s'},
                                            ydot={'shape': (num_nodes,), 'units': 'm/s'},
                                            has_diag_partials=True)

        phase = dm.Phase(ode_class=ode, transcription=t)
        # upgrade_doc: end declare_rate_source
        traj = dm.Trajectory()
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=solve_segments,
                        rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=solve_segments,
                        rate_source='ydot')

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=solve_segments,
                        rate_source='vdot')

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', dynamic=False)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['traj0.phase0.parameters:g'] = 9.80665

        dm.run_problem(p, run_driver=run_driver, simulate=True)

        return p

    def run_asserts(self):

        for db in ['dymos_solution.db', 'dymos_simulation.db']:
            p = om.CaseReader(db).get_case('final')

            t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
            tf = p.get_val('traj0.phase0.timeseries.time')[-1]

            x0 = p.get_val('traj0.phase0.timeseries.states:x')[0]
            xf = p.get_val('traj0.phase0.timeseries.states:x')[-1]

            y0 = p.get_val('traj0.phase0.timeseries.states:y')[0]
            yf = p.get_val('traj0.phase0.timeseries.states:y')[-1]

            v0 = p.get_val('traj0.phase0.timeseries.states:v')[0]
            vf = p.get_val('traj0.phase0.timeseries.states:v')[-1]

            g = p.get_val('traj0.phase0.timeseries.parameters:g')[0]

            thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1]

            assert_near_equal(t_initial, 0.0)
            assert_near_equal(x0, 0.0)
            assert_near_equal(y0, 10.0)
            assert_near_equal(v0, 0.0)

            assert_near_equal(tf, 1.8016, tolerance=0.01)
            assert_near_equal(xf, 10.0, tolerance=0.01)
            assert_near_equal(yf, 5.0, tolerance=0.01)
            assert_near_equal(vf, 9.902, tolerance=0.01)
            assert_near_equal(g, 9.80665, tolerance=0.01)

            assert_near_equal(thetaf, 100.12, tolerance=0.01)

    def test_ex_brachistochrone_radau_uncompressed(self):
        self._make_problem(transcription='radau-ps', compressed=False)
        self.run_asserts()

    def test_ex_brachistochrone_gl_uncompressed(self):
        self._make_problem(transcription='gauss-lobatto', compressed=False)
        self.run_asserts()
