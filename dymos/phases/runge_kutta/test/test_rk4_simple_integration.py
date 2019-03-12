from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_rel_error

from dymos import RungeKuttaPhase
from dymos.phases.runge_kutta.test.rk_test_ode import TestODE, _test_ode_solution


class TestRK4SimpleIntegration(unittest.TestCase):

    def test_simple_integration_forward(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=200,
                            method='rk4',
                            ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        expected = _test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])
        assert_rel_error(self, p['phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_simple_integration_forward_connected_initial(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=200,
                            method='rk4',
                            ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=False, connected_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        # The initial guess of states at the segment boundaries
        p['phase0.states:y'] = 0.0

        # The initial value of the states from which the integration proceeds
        p['phase0.initial_states:y'] = 0.5

        p.run_model()

        expected = _test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])
        assert_rel_error(self, p['phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_simple_integration_forward_connected_initial_fixed_initial(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=200,
                            method='rk4',
                            ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True, connected_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        with self.assertRaises(ValueError) as e:
            p.setup(check=True, force_alloc_complex=True)
        expected = "Cannot specify 'fix_initial=True' and specify 'connected_initial=True' for " \
                   "state y in phase phase0."
        self.assertEqual(str(e.exception), expected)

    def test_simple_integration_backward(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      RungeKuttaPhase(num_segments=4, method='rk4',
                                                      ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 2.0
        p['phase0.t_duration'] = -2.0

        p['phase0.states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])).T
        assert_rel_error(self, p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)

    def test_simple_integration_backward_connected_initial(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0',
                                      RungeKuttaPhase(num_segments=4, method='rk4',
                                                      ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', connected_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 2.0
        p['phase0.t_duration'] = -2.0

        # The initial guess of state values at the segment boundaries
        p['phase0.states:y'] = 0

        # The initial value of the states from which the integration proceeds
        p['phase0.initial_states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])).T
        assert_rel_error(self, p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)


if __name__ == '__main__':
    unittest.main()
