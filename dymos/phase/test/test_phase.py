import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


import matplotlib.pyplot as plt
plt.switch_backend('Agg')


OPTIMIZER = 'SLSQP'


class _A(object):
    pass


class _B(om.Group):
    pass


class _C(om.ExplicitComponent):
    pass


class _D(om.ExplicitComponent):
    ode_options = None


@use_tempdirs
class TestPhaseBase(unittest.TestCase):

    def test_invalid_ode_wrong_class(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=_A,
                         transcription=dm.GaussLobatto(num_segments=20, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=False)

        phase.add_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        expected = 'If given as a class, ode_class must be derived from openmdao.core.System.'
        self.assertEqual(expected, str(e.exception))

    def test_invalid_ode_instance(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=_A(),
                         transcription=dm.GaussLobatto(num_segments=20, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=False)

        phase.add_parameter('g', units='m/s**2', opt=True, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        expected = 'ode_class must be given as a callable object that returns an object derived ' \
                   'from openmdao.core.System, or as a class derived from openmdao.core.System.'
        self.assertEqual(expected, str(e.exception))

    def test_add_existing_parameter_as_parameter(self):

        p = dm.Phase(ode_class=_A,
                     transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.add_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_parameter('theta')

        expected = 'theta has already been added as a parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_control_as_parameter(self):

        p = dm.Phase(ode_class=BrachistochroneODE,
                     transcription=dm.GaussLobatto(num_segments=8, order=3))

        p.add_control('theta')

        with self.assertRaises(ValueError) as e:
            p.add_parameter('theta')

        expected = 'theta has already been added as a control.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_options_nonoptimal_param(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=16, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)

        phase.add_state('y', fix_initial=True, fix_final=False)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, lower=5, upper=10,
                            ref0=5, ref=10, scaler=1, adder=0)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.setup(check=False)

        print('\n'.join([str(ww.message) for ww in w]))

        expected = 'Invalid options for non-optimal parameter \'g\' in phase \'phase0\': ' \
                   'lower, upper, scaler, adder, ref, ref0'

        self.assertIn(expected, [str(ww.message) for ww in w])

    def test_invalid_options_nonoptimal_control(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=False,
                          units='deg', lower=0.01, upper=179.9, scaler=1, ref=1, ref0=0)

        phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

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

    def test_objective_parameter_gl(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p['phase0.t_duration'], 10, tolerance=1.0E-3)

    def test_objective_parameter_radau(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=20, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(4, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('g')

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p['phase0.t_duration'], 10, tolerance=1.0E-3)

    def test_control_boundary_constraint_gl(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        phase.add_state('x', fix_initial=True, fix_final=False)

        phase.add_state('y', fix_initial=True, fix_final=False)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665)

        phase.add_boundary_constraint('theta', loc='final', lower=90.0, upper=90.0, units='deg')

        # Minimize time at the end of the phase
        phase.add_objective('time')

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 8

        p.run_driver()

        assert_near_equal(p.get_val('phase0.timeseries.controls:theta', units='deg')[-1], 90.0)

    def test_control_rate_boundary_constraint_gl(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.1, 10))

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)

        phase.add_state('y', fix_initial=True, fix_final=False)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, units='deg',
                          lower=0.01, upper=179.9)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665)

        phase.add_boundary_constraint('theta_rate', loc='final', equals=0.0, units='deg/s')

        # Minimize time at the end of the phase
        phase.add_objective('time')

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 8

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

        assert_near_equal(p.get_val('phase0.timeseries.control_rates:theta_rate')[-1], 0,
                          tolerance=1.0E-6)

    def test_control_rate2_boundary_constraint_gl(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)

        phase.add_state('y', fix_initial=True, fix_final=False)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, rate2_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665)

        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0.0, units='deg/s**2')

        # Minimize time at the end of the phase
        phase.add_objective('time')

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 8

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

        assert_near_equal(p.get_val('phase0.timeseries.control_rates:theta_rate2')[-1], 0,
                          tolerance=1.0E-6)

    def test_parameter_boundary_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=20,
                                                       order=3,
                                                       compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, rate2_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

        # We'll let g vary, but make sure it hits the desired value.
        # It's a static parameter, so it shouldn't matter whether we enforce it
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

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 5

        p.run_driver()

        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016,
                          tolerance=1.0E-4)
        assert_near_equal(p.get_val('phase0.timeseries.parameters:g')[0], 9.80665,
                          tolerance=1.0E-6)
        assert_near_equal(p.get_val('phase0.timeseries.parameters:g')[-1], 9.80665,
                          tolerance=1.0E-6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
