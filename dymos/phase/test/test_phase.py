from __future__ import print_function, division, absolute_import

import unittest
import warnings

import openmdao.api as om

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.assert_utils import assert_rel_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OPTIMIZER = 'SLSQP'
SHOW_PLOTS = False


class _A(object):
    pass


class _B(om.Group):
    pass


class _C(om.ExplicitComponent):
    pass


class _D(om.ExplicitComponent):
    ode_options = None


class TestPhaseBase(unittest.TestCase):

    def test_invalid_ode_wrong_class(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=_A,
                         transcription=dm.GaussLobatto(num_segments=20, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=False)

        phase.add_design_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        self.assertEqual(str(e.exception), 'ode_class must be derived from openmdao.core.System.')

    def test_invalid_ode_instance(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=_A(),
                         transcription=dm.GaussLobatto(num_segments=20, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=False)

        phase.add_design_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        self.assertEqual(str(e.exception), 'ode_class must be a class, not an instance.')

    def test_add_existing_design_parameter_as_design_parameter(self):

        p = dm.Phase(ode_class=_A,
                     transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.add_design_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as a design parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_control_as_design_parameter(self):

        p = dm.Phase(ode_class=BrachistochroneODE,
                     transcription=dm.GaussLobatto(num_segments=8, order=3))

        p.add_control('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as a control.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_input_parameter_as_design_parameter(self):

        p = dm.Phase(ode_class=_A,
                     transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.add_input_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as an input parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_options_nonoptimal_design_param(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=16, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=False)

        phase.add_design_parameter('g', units='m/s**2', opt=False, lower=5, upper=10,
                                   ref0=5, ref=10, scaler=1, adder=0)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.setup(check=False)

        print('\n'.join([str(ww.message) for ww in w]))

        expected = 'Invalid options for non-optimal design_parameter \'g\' in phase \'phase0\': ' \
                   'lower, upper, scaler, adder, ref, ref0'

        self.assertIn(expected, [str(ww.message) for ww in w])

    def test_add_existing_design_parameter_as_input_parameter(self):
        p = dm.Phase(ode_class=_A,
                     transcription=dm.GaussLobatto(num_segments=14, order=3, compressed=True))

        p.add_design_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as a design parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_control_as_input_parameter(self):

        p = dm.Phase(ode_class=_A,
                     transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.add_control('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as a control.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_input_parameter_as_input_parameter(self):

        p = dm.Phase(ode_class=_A,
                     transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.add_input_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as an input parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_options_nonoptimal_control(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=False,
                          units='deg', lower=0.01, upper=179.9, scaler=1, ref=1, ref0=0)

        phase.add_design_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.setup(check=True)

        expected = 'Invalid options for non-optimal control \'theta\' in phase \'phase0\': ' \
                   'lower, upper, scaler, ref, ref0'

        self.assertIn(expected, [str(ww.message) for ww in w])

    def test_invalid_boundary_loc(self):

        p = dm.Phase(ode_class=BrachistochroneODE,
                     transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        with self.assertRaises(ValueError) as e:
            p.add_boundary_constraint('x', loc='foo')

        expected = 'Invalid boundary constraint location "foo". Must be "initial" or "final".'
        self.assertEqual(str(e.exception), expected)

    def test_objective_design_parameter_gl(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

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

        p.model.linear_solver = om.DirectSolver()
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
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=20, order=3, compressed=True))

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
        p.model.linear_solver = om.DirectSolver()
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
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.1, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, rate2_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta', loc='final', lower=90.0, upper=90.0, units='deg')

        # Minimize time at the end of the phase
        phase.add_objective('time')

        p.model.linear_solver = om.DirectSolver()
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

    def test_control_rate_boundary_constraint_gl(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.1, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, rate2_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta_rate', loc='final', equals=0.0, units='deg/s')

        # Minimize time at the end of the phase
        phase.add_objective('time')

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 8

        p.run_driver()

        import matplotlib.pyplot as plt

        plt.plot(p.get_val('phase0.timeseries.states:x'),
                 p.get_val('phase0.timeseries.states:y'), 'ko')

        plt.figure()

        plt.plot(p.get_val('phase0.timeseries.time'),
                 p.get_val('phase0.timeseries.controls:theta'), 'ro')

        plt.plot(p.get_val('phase0.timeseries.time'),
                 p.get_val('phase0.timeseries.control_rates:theta_rate'), 'bo')

        plt.plot(p.get_val('phase0.timeseries.time'),
                 p.get_val('phase0.timeseries.control_rates:theta_rate2'), 'go')
        plt.show()

        assert_rel_error(self, p.get_val('phase0.timeseries.control_rates:theta_rate')[-1], 0,
                         tolerance=1.0E-6)

    def test_control_rate2_boundary_constraint_gl(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.1, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, rate2_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0.0, units='deg/s**2')

        # Minimize time at the end of the phase
        phase.add_objective('time')

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 8

        p.run_driver()

        plt.plot(p.get_val('phase0.timeseries.states:x'),
                 p.get_val('phase0.timeseries.states:y'), 'ko')

        plt.figure()

        plt.plot(p.get_val('phase0.timeseries.time'),
                 p.get_val('phase0.timeseries.controls:theta'), 'ro')

        plt.plot(p.get_val('phase0.timeseries.time'),
                 p.get_val('phase0.timeseries.control_rates:theta_rate'), 'bo')

        plt.plot(p.get_val('phase0.timeseries.time'),
                 p.get_val('phase0.timeseries.control_rates:theta_rate2'), 'go')
        plt.show()

        assert_rel_error(self, p.get_val('phase0.timeseries.control_rates:theta_rate2')[-1], 0,
                         tolerance=1.0E-6)

    def test_design_parameter_boundary_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # We'll let g vary, but make sure it hits the desired value.
        # It's a static design parameter, so it shouldn't matter whether we enforce it
        # at the start or the end of the phase, so here we'll do both.
        # Note if we make these equality constraints, some optimizers (SLSQP) will
        # see the problem as infeasible.
        phase.add_boundary_constraint('g', loc='initial', units='m/s**2', upper=9.80665)
        phase.add_boundary_constraint('g', loc='final', units='m/s**2', upper=9.80665)

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
        p['phase0.design_parameters:g'] = 5

        p.run_driver()

        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016,
                         tolerance=1.0E-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.design_parameters:g')[0], 9.80665,
                         tolerance=1.0E-6)
        assert_rel_error(self, p.get_val('phase0.timeseries.design_parameters:g')[-1], 9.80665,
                         tolerance=1.0E-6)


if __name__ == '__main__':
    unittest.main()
