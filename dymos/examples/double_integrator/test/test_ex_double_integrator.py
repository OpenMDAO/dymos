from __future__ import print_function, absolute_import, division

import itertools
import sys
import unittest

from numpy.testing import assert_almost_equal

from parameterized import parameterized

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, IndepVarComp

from dymos import Phase
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE
import dymos.examples.double_integrator.ex_double_integrator as ex_double_integrator

PY_MAJOR, PY_MINOR, _, _, _ = sys.version_info
PY_VERSION = PY_MAJOR + 0.1 * PY_MINOR


class TestDoubleIntegratorExample(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['dense', 'csc'],  # jacobian
                          ['compressed', 'uncompressed'],  # compressed transcription
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1],
                                                                          p.args[2]])
    )
    def test_ex_double_integrator(self, transcription='radau-ps', jacobian='csc',
                                  compressed='compressed'):
        ex_double_integrator.SHOW_PLOTS = False
        p = ex_double_integrator.double_integrator_direct_collocation(
            transcription, top_level_jacobian=jacobian, compressed=compressed == 'compressed')

        x0 = p.model.phase0.get_values('x')[0]
        xf = p.model.phase0.get_values('x')[-1]

        v0 = p.model.phase0.get_values('v')[0]
        vf = p.model.phase0.get_values('v')[-1]

        assert_almost_equal(x0, 0.0)
        assert_almost_equal(xf, 0.25)

        assert_almost_equal(v0, 0.0)
        assert_almost_equal(vf, 0.0)

    def test_ex_double_integrator_input_times(self, transcription='radau-ps',
                                              compressed=True):
        """
        Tests that externally connected t_initial and t_duration function as expected.
        """

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        times_ivc = p.model.add_subsystem('times_ivc', IndepVarComp(),
                                          promotes_outputs=['t0', 'tp'])
        times_ivc.add_output(name='t0', val=0.0, units='s')
        times_ivc.add_output(name='tp', val=1.0, units='s')

        phase = Phase(transcription,
                      ode_class=DoubleIntegratorODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=compressed)

        p.model.add_subsystem('phase0', phase)

        p.model.connect('t0', 'phase0.t_initial')
        p.model.connect('tp', 'phase0.t_duration')

        phase.set_time_options(input_initial=True, input_duration=True)

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('v', fix_initial=True, fix_final=True)

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(check=True)

        p['t0'] = 0.0
        p['tp'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()

    @unittest.skipIf(PY_VERSION < 3.3, 'assertWarns not available in this version of Python')
    def test_ex_double_integrator_input_and_fixed_times_warns(self, transcription='radau-ps'):
        """
        Tests that time optimization options cause a ValueError to be raised when t_initial and
        t_duration are connected to external sources.
        """

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        times_ivc = p.model.add_subsystem('times_ivc', IndepVarComp(),
                                          promotes_outputs=['t0', 'tp'])
        times_ivc.add_output(name='t0', val=0.0, units='s')
        times_ivc.add_output(name='tp', val=1.0, units='s')

        phase = Phase(transcription,
                      ode_class=DoubleIntegratorODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        p.model.connect('t0', 'phase0.t_initial')
        p.model.connect('tp', 'phase0.t_duration')

        with self.assertWarns(RuntimeWarning) as w:
            phase.set_time_options(input_initial=True, fix_initial=True, input_duration=True,
                                   fix_duration=True)

        self.assertTrue(len(w.warnings) == 2,
                        'Expected 2 warnings, got {0}'.format(len(w.warnings)))

        expected = 'Phase "phase0" initial time is an externally-connected input, therefore ' \
                   'fix_initial has no effect.'
        self.assertEqual(str(w.warnings[0].message), expected)
        expected = 'Phase "phase0" time duration is an externally-connected input, ' \
                   'therefore fix_duration has no effect.'
        self.assertEqual(str(w.warnings[1].message), expected)

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('v', fix_initial=True, fix_final=True)

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(check=True)

        p['t0'] = 0.0
        p['tp'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()

    @unittest.skipIf(PY_VERSION < 3.3, 'assertWarns not available in this version of Python')
    def test_ex_double_integrator_input_times_warns(self, transcription='radau-ps'):
        """
        Tests that time optimization options cause a ValueError to be raised when t_initial and
        t_duration are connected to external sources.
        """

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        times_ivc = p.model.add_subsystem('times_ivc', IndepVarComp(),
                                          promotes_outputs=['t0', 'tp'])
        times_ivc.add_output(name='t0', val=0.0, units='s')
        times_ivc.add_output(name='tp', val=1.0, units='s')

        phase = Phase(transcription,
                      ode_class=DoubleIntegratorODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        p.model.connect('t0', 'phase0.t_initial')
        p.model.connect('tp', 'phase0.t_duration')

        with self.assertWarns(RuntimeWarning) as cm:
            phase.set_time_options(input_initial=True, initial_bounds=(-5, 5), initial_ref0=1,
                                   initial_ref=10, initial_adder=0, initial_scaler=1,
                                   input_duration=True, duration_bounds=(-5, 5), duration_ref0=1,
                                   duration_ref=10, duration_adder=0, duration_scaler=1)

        self.assertTrue(len(cm.warnings) == 2,
                        msg='Expected 2 warnings, got {0}'.format(len(cm.warnings)))

        self.assertEqual(str(cm.warnings[0].message),
                         'Phase time options have no effect because input_initial=True for phase '
                         '"phase0": initial_bounds, initial_scaler, initial_adder, initial_ref, '
                         'initial_ref0')

        self.assertEqual(str(cm.warnings[1].message),
                         'Phase time options have no effect because input_duration=True for phase '
                         '"phase0": duration_bounds, duration_scaler, duration_adder, duration_ref,'
                         ' duration_ref0')

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('v', fix_initial=True, fix_final=True)

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(check=True)

        p['t0'] = 0.0
        p['tp'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()

    @unittest.skipIf(PY_VERSION < 3.3, 'assertWarns not available in this version of Python')
    def test_ex_double_integrator_fixed_times_warns(self, transcription='radau-ps'):
        """
        Tests that time optimization options cause a ValueError to be raised when t_initial and
        t_duration are connected to external sources.
        """

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase(transcription,
                      ode_class=DoubleIntegratorODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        with self.assertWarns(RuntimeWarning) as cm:
            phase.set_time_options(fix_initial=True, initial_bounds=(-5, 5), initial_ref0=1,
                                   initial_ref=10, initial_adder=0, initial_scaler=1,
                                   fix_duration=True, duration_bounds=(-5, 5), duration_ref0=1,
                                   duration_ref=10, duration_adder=0, duration_scaler=1)

        self.assertTrue(len(cm.warnings) == 2,
                        msg='Expected 2 warnings, got {0}'.format(len(cm.warnings)))

        self.assertEqual(str(cm.warnings[0].message),
                         'Phase time options have no effect because fix_initial=True for phase '
                         '"phase0": initial_bounds, initial_scaler, initial_adder, initial_ref, '
                         'initial_ref0')

        self.assertEqual(str(cm.warnings[1].message),
                         'Phase time options have no effect because fix_duration=True for phase '
                         '"phase0": duration_bounds, duration_scaler, duration_adder, duration_ref,'
                         ' duration_ref0')

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('v', fix_initial=True, fix_final=True)

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()

    @unittest.skipIf(PY_VERSION < 3.3, 'assertWarns not available in this version of Python')
    def test_ex_double_integrator_deprecated_time_options(self, transcription='radau-ps'):
        """
        Tests that time optimization options cause a ValueError to be raised when t_initial and
        t_duration are connected to external sources.
        """

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase(transcription,
                      ode_class=DoubleIntegratorODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        with self.assertWarns(RuntimeWarning) as cm:
            phase.set_time_options(opt_initial=False, initial_bounds=(-5, 5), initial_ref0=1,
                                   initial_ref=10, initial_adder=0, initial_scaler=1,
                                   opt_duration=False, duration_bounds=(-5, 5), duration_ref0=1,
                                   duration_ref=10, duration_adder=0, duration_scaler=1)

        for w in cm.warnings:
            print(w.message)

        self.assertTrue(len(cm.warnings) == 4,
                        msg='Expected 4 warnings, got {0}'.format(len(cm.warnings)))

        self.assertEqual(str(cm.warnings[0].message), 'opt_initial has been deprecated in favor of '
                                                      'fix_initial, which has the opposite meaning.'
                                                      ' If the user desires to input the initial '
                                                      'phase time from an exterior source, set '
                                                      'input_initial=True.')

        self.assertEqual(str(cm.warnings[1].message), 'opt_duration has been deprecated in favor '
                                                      'of fix_duration, which has the opposite '
                                                      'meaning. If the user desires to input the '
                                                      'phase duration from an exterior source, '
                                                      'set input_duration=True.')

        self.assertEqual(str(cm.warnings[2].message),
                         'Phase time options have no effect because fix_initial=True for phase '
                         '"phase0": initial_bounds, initial_scaler, initial_adder, initial_ref, '
                         'initial_ref0')

        self.assertEqual(str(cm.warnings[3].message),
                         'Phase time options have no effect because fix_duration=True for phase '
                         '"phase0": duration_bounds, duration_scaler, duration_adder, duration_ref,'
                         ' duration_ref0')

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('v', fix_initial=True, fix_final=True)

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()
