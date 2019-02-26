from __future__ import print_function, division, absolute_import

import os
import os.path
import unittest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import ExplicitComponent, Group

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


OPTIMIZER = 'SLSQP'
SHOW_PLOTS = False


class _A(object):
    pass


class _B(Group):
    pass


class _C(ExplicitComponent):
    pass


class _D(ExplicitComponent):
    ode_options = None


class TestPhaseBase(unittest.TestCase):

    def test_invalid_ode_class_wrong_class(self):

        with self.assertRaises(ValueError) as e:
            phase = Phase('gauss-lobatto',
                          ode_class=_A,
                          num_segments=8,
                          transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class must be derived from openmdao.core.System.')

    def test_invalid_ode_class_no_metadata(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=_B,
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_no_metadata2(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=_C,
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_invalid_metadata(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=_D,
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_instance(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE(),
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class must be a class, not an instance.')

    def test_invalid_design_parameter_name(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with self.assertRaises(ValueError) as e:

            p.add_design_parameter('foo')

        expected = 'foo is not a controllable parameter in the ODE system.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_design_parameter_as_design_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_design_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as a design parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_control_as_design_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_control('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as a control.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_input_parameter_as_design_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_input_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as an input parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_options_nonoptimal_design_param(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.add_design_parameter('g', opt=False, lower=5, upper=10, ref0=5, ref=10,
                                   scaler=1, adder=0)

        expected = 'Invalid options for non-optimal design parameter "g":' \
                   'lower, upper, scaler, adder, ref, ref0'

        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message), expected)

    def test_invalid_input_parameter_name(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with self.assertRaises(ValueError) as e:

            p.add_input_parameter('foo')

        expected = 'foo is not a controllable parameter in the ODE system.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_design_parameter_as_input_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_design_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as a design parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_control_as_input_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_control('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as a control.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_input_parameter_as_input_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_input_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as an input parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_options_nonoptimal_control(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.add_control('theta', opt=False, lower=5, upper=10, ref0=5, ref=10,
                          scaler=1, adder=0)

        expected = 'Invalid options for non-optimal control "theta":' \
                   'lower, upper, scaler, adder, ref, ref0'

        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message), expected)

    def test_invalid_boundary_loc(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with self.assertRaises(ValueError) as e:
            p.add_boundary_constraint('x', loc='foo')

        expected = 'Invalid boundary constraint location "foo". Must be "initial" or "final".'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_set_options(self):

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        with self.assertRaises(ValueError) as e:
            phase.set_state_options('x', fix_initial=False, fix_final=False,
                                    solve_segments=False, solve_continuity=True)

        msg = "The 'solve_continuity' option can only be used when 'solve_segments' is True."
        self.assertEqual(str(e.exception), msg)

    def test_objective_design_parameter_gl(self):
        from openmdao.api import Problem, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error

        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

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

        assert_rel_error(self, p['phase0.t_duration'], 10, tolerance=1.0E-3)

    def test_objective_design_parameter_radau(self):
        from openmdao.api import Problem, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error

        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('radau-ps',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.options['assembled_jac_type'] = 'csc'
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

        assert_rel_error(self, p['phase0.t_duration'], 10, tolerance=1.0E-3)

    def test_control_boundary_constraint_gl(self):
        from openmdao.api import Problem, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error

        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.1, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=True, val=9.80665)

        phase.add_boundary_constraint('theta', loc='final', lower=90.0, upper=90.0, units='deg')
        phase.add_boundary_constraint('theta_rate', loc='final', equals=0.0, units='deg/s')
        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0.0, units='deg/s**2')
        phase.add_boundary_constraint('g', loc='initial', equals=9.80665, units='m/s**2')

        # Minimize time at the end of the phase
        phase.add_objective('time')

        p.model.linear_solver = DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 8

        p.run_driver()

        assert_rel_error(self, p.get_val('phase0.timeseries.controls:theta', units='deg')[-1], 90.0)
        assert_rel_error(self, p.get_val('phase0.timeseries.control_rates:theta_rate')[-1], 0,
                         tolerance=1.0E-6)
        assert_rel_error(self, p.get_val('phase0.timeseries.control_rates:theta_rate2')[-1], 0,
                         tolerance=1.0E-6)
        assert_rel_error(self, p.get_val('phase0.timeseries.design_parameters:g')[0], 9.80665,
                         tolerance=1.0E-6)


if __name__ == '__main__':
    unittest.main()
