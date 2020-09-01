import os
import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestPhaseTimeOptions(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_fixed_time_invalid_options(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, fix_duration=True,
                               initial_bounds=(1.0, 5.0), initial_adder=0.0,
                               initial_scaler=1.0, initial_ref0=0.0,
                               initial_ref=1.0, duration_bounds=(1.0, 5.0),
                               duration_adder=0.0, duration_scaler=1.0,
                               duration_ref0=0.0, duration_ref=1.0)

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta',
                                     order=1,
                                     units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = om.DirectSolver()

        expected_msg0 = 'Phase time options have no effect because fix_initial=True or ' \
                        'input_initial=True for phase \'phase0\': initial_bounds, initial_scaler, ' \
                        'initial_adder, initial_ref, initial_ref0'

        expected_msg1 = 'Phase time options have no effect because fix_duration=True or ' \
                        'input_duration=True for phase \'phase0\': duration_bounds, ' \
                        'duration_scaler, duration_adder, duration_ref, duration_ref0'

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)

        self.assertIn(expected_msg0, [str(w.message) for w in ctx])
        self.assertIn(expected_msg1, [str(w.message) for w in ctx])

    def test_initial_val_and_final_val_stick(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=False, fix_duration=False,
                               initial_val=0.01, duration_val=1.9)

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        assert_near_equal(p['phase0.t_initial'], 0.01)
        assert_near_equal(p['phase0.t_duration'], 1.9)

    def test_input_and_fixed_times_warns(self):
        """
        Tests that time optimization options cause a ValueError to be raised when t_initial and
        t_duration are connected to external sources.
        """
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(input_initial=True, fix_initial=True,
                               input_duration=True, fix_duration=True)

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = om.DirectSolver()

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)

        expected_msg0 = 'Phase \'phase0\' initial time is an externally-connected input, therefore ' \
                        'fix_initial has no effect.'

        expected_msg1 = 'Phase \'phase0\' time duration is an externally-connected input, ' \
                        'therefore fix_duration has no effect.'

        self.assertIn(expected_msg0, [str(w.message) for w in ctx])
        self.assertIn(expected_msg1, [str(w.message) for w in ctx])

    def test_input_time_invalid_options(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(input_initial=True, input_duration=True,
                               initial_bounds=(1.0, 5.0), initial_adder=0.0,
                               initial_scaler=1.0, initial_ref0=0.0,
                               initial_ref=1.0, duration_bounds=(1.0, 5.0),
                               duration_adder=0.0, duration_scaler=1.0,
                               duration_ref0=0.0, duration_ref=1.0)

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = om.DirectSolver()

        expected_msg0 = 'Phase time options have no effect because fix_initial=True or ' \
                        'input_initial=True for phase \'phase0\': initial_bounds, ' \
                        'initial_scaler, initial_adder, initial_ref, initial_ref0'

        expected_msg1 = 'Phase time options have no effect because fix_duration=True or ' \
                        'input_duration=True for phase \'phase0\': duration_bounds, ' \
                        'duration_scaler, duration_adder, duration_ref, duration_ref0'

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)

        self.assertIn(expected_msg0, [str(w.message) for w in ctx])
        self.assertIn(expected_msg1, [str(w.message) for w in ctx])

    def test_unbounded_time(self):
            p = om.Problem(model=om.Group())

            p.driver = om.ScipyOptimizeDriver()
            p.driver.declare_coloring()

            phase = dm.Phase(ode_class=BrachistochroneODE,
                             transcription=dm.GaussLobatto(num_segments=8, order=3))

            p.model.add_subsystem('phase0', phase)

            phase.set_time_options(fix_initial=False, fix_duration=False)

            phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                            units=BrachistochroneODE.states['x']['units'],
                            fix_initial=True, fix_final=True, solve_segments=False)

            phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                            units=BrachistochroneODE.states['y']['units'],
                            fix_initial=True, fix_final=True, solve_segments=False)

            phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                            units=BrachistochroneODE.states['v']['units'],
                            fix_initial=True, fix_final=False, solve_segments=False)

            phase.add_control('theta', units='deg', lower=0.01, upper=179.9)

            phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

            # Minimize time at the end of the phase
            phase.add_objective('time', loc='final', scaler=10)

            phase.add_boundary_constraint('time', loc='initial', equals=0)

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

            self.assertTrue(p.driver.result.success,
                            msg='Brachistochrone with outbounded times has failed')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
