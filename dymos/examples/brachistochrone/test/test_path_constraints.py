from __future__ import print_function, absolute_import, division

import unittest


class TestBrachistochronePathConstraints(unittest.TestCase):

    def test_control_rate_path_constraint_gl(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'])
        phase.add_state('y', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'])
        phase.add_state('v', fix_initial=True,
                        rate_source=BrachistochroneODE.states['v']['rate_source'],
                        targets=BrachistochroneODE.states['v']['targets'],
                        units=BrachistochroneODE.states['v']['units'])

        phase.add_control('theta', units='deg',
                          targets=BrachistochroneODE.parameters['theta']['targets'],
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_design_parameter('g',
                                   targets=BrachistochroneODE.parameters['g']['targets'],
                                   units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate', lower=0, upper=100, units='deg/s')

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

    def test_control_rate2_path_constraint_gl(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=5))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'])
        phase.add_state('y', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'])
        phase.add_state('v', fix_initial=True,
                        rate_source=BrachistochroneODE.states['v']['rate_source'],
                        targets=BrachistochroneODE.states['v']['targets'],
                        units=BrachistochroneODE.states['v']['units'])

        phase.add_control('theta', units='deg',
                          targets=BrachistochroneODE.parameters['theta']['targets'],
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_design_parameter('g',
                                   targets=BrachistochroneODE.parameters['g']['targets'],
                                   units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate2', lower=-200, upper=200, units='rad/s**2')

        p.model.linear_solver = om.DirectSolver()
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

    def test_control_rate_path_constraint_radau(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10,
                                                compressed=False))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'])
        phase.add_state('y', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'])
        phase.add_state('v', fix_initial=True,
                        rate_source=BrachistochroneODE.states['v']['rate_source'],
                        targets=BrachistochroneODE.states['v']['targets'],
                        units=BrachistochroneODE.states['v']['units'])

        phase.add_control('theta', units='deg',
                          targets=BrachistochroneODE.parameters['theta']['targets'],
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_design_parameter('g',
                                   targets=BrachistochroneODE.parameters['g']['targets'],
                                   units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate', lower=0, upper=100, units='deg/s')

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0.9, 101.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

    def test_control_rate2_path_constraint_radau(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10,
                                                compressed=False))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'])
        phase.add_state('y', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'])
        phase.add_state('v', fix_initial=True,
                        rate_source=BrachistochroneODE.states['v']['rate_source'],
                        targets=BrachistochroneODE.states['v']['targets'],
                        units=BrachistochroneODE.states['v']['units'])

        phase.add_control('theta', units='deg',
                          targets=BrachistochroneODE.parameters['theta']['targets'],
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_design_parameter('g',
                                   targets=BrachistochroneODE.parameters['g']['targets'],
                                   units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate2', lower=-200, upper=200, units='rad/s**2')

        p.model.linear_solver = om.DirectSolver()
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)
