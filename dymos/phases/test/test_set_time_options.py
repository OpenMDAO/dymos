from __future__ import print_function, division, absolute_import

import os
import unittest
import warnings

from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE


class TestPhaseTimeOptions(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_fixed_time_invalid_options(self):
        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, fix_duration=True,
                               initial_bounds=(1.0, 5.0), initial_adder=0.0,
                               initial_scaler=1.0, initial_ref0=0.0,
                               initial_ref=1.0, duration_bounds=(1.0, 5.0),
                               duration_adder=0.0, duration_scaler=1.0,
                               duration_ref0=0.0, duration_ref=1.0)

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = DirectSolver()

        expected_msg0 = 'Phase time options have no effect because fix_initial=True for ' \
                        'phase \'phase0\': initial_bounds, initial_scaler, initial_adder, ' \
                        'initial_ref, initial_ref0'

        expected_msg1 = 'Phase time options have no effect because fix_duration=True for' \
                        ' phase \'phase0\': duration_bounds, duration_scaler, ' \
                        'duration_adder, duration_ref, duration_ref0'

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)

        self.assertEqual(len(ctx), 2,
                         msg='setup failed to raise two warnings')
        self.assertEqual(str(ctx[0].message), expected_msg0)
        self.assertEqual(str(ctx[1].message), expected_msg1)

    def test_initial_val_and_final_val_stick(self):
        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=False, fix_duration=False,
                               initial_val=0.01, duration_val=1.9)

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = DirectSolver()
        p.setup(check=True)

        assert_rel_error(self, p['phase0.t_initial'], 0.01)
        assert_rel_error(self, p['phase0.t_duration'], 1.9)

    def test_ex_double_integrator_input_and_fixed_times_warns(self, transcription='radau-ps'):
        """
        Tests that time optimization options cause a ValueError to be raised when t_initial and
        t_duration are connected to external sources.
        """
        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(input_initial=True, fix_initial=True,
                               input_duration=True, fix_duration=True)

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = DirectSolver()

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)

        self.assertTrue(len(ctx) == 2,
                        'Expected 2 warnings, got {0}'.format(len(ctx)))

        expected = 'Phase \'phase0\' initial time is an externally-connected input, therefore ' \
                   'fix_initial has no effect.'
        self.assertEqual(str(ctx[0].message), expected)
        expected = 'Phase \'phase0\' time duration is an externally-connected input, ' \
                   'therefore fix_duration has no effect.'
        self.assertEqual(str(ctx[1].message), expected)

    def test_input_time_invalid_options(self):
        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(input_initial=True, input_duration=True,
                               initial_bounds=(1.0, 5.0), initial_adder=0.0,
                               initial_scaler=1.0, initial_ref0=0.0,
                               initial_ref=1.0, duration_bounds=(1.0, 5.0),
                               duration_adder=0.0, duration_scaler=1.0,
                               duration_ref0=0.0, duration_ref=1.0)

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('time', loc='initial', equals=0)

        p.model.linear_solver = DirectSolver()

        expected_msg0 = 'Phase time options have no effect because fix_initial=True for ' \
                        'phase \'phase0\': initial_bounds, initial_scaler, initial_adder, ' \
                        'initial_ref, initial_ref0'

        expected_msg1 = 'Phase time options have no effect because fix_duration=True for' \
                        ' phase \'phase0\': duration_bounds, duration_scaler, ' \
                        'duration_adder, duration_ref, duration_ref0'

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)

        self.assertEqual(len(ctx), 2,
                         msg='setup failed to raise two warnings')
        self.assertEqual(str(ctx[0].message), expected_msg0)
        self.assertEqual(str(ctx[1].message), expected_msg1)

    def test_unbounded_time(self):
            p = Problem(model=Group())

            p.driver = ScipyOptimizeDriver()
            p.driver.options['dynamic_simul_derivs'] = True

            phase = Phase('gauss-lobatto',
                          ode_class=BrachistochroneODE,
                          num_segments=8,
                          transcription_order=3)

            p.model.add_subsystem('phase0', phase)

            phase.set_time_options(fix_initial=False, fix_duration=False)

            phase.set_state_options('x', fix_initial=True, fix_final=True)
            phase.set_state_options('y', fix_initial=True, fix_final=True)
            phase.set_state_options('v', fix_initial=True, fix_final=False)

            phase.add_control('theta', continuity=True, rate_continuity=True,
                              units='deg', lower=0.01, upper=179.9)

            phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

            # Minimize time at the end of the phase
            phase.add_objective('time', loc='final', scaler=10)

            phase.add_boundary_constraint('time', loc='initial', equals=0)

            p.model.linear_solver = DirectSolver()
            p.setup(check=True)

            p['phase0.t_initial'] = 0.0
            p['phase0.t_duration'] = 2.0

            p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
            p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
            p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
            p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
            p['phase0.design_parameters:g'] = 9.80665

            p.run_driver()

            self.assertTrue(p.driver.result.success,
                            msg='Brachistochrone with outbounded times has failed')


if __name__ == '__main__':
    unittest.main()
