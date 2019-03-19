from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_rel_error

from dymos.phases.solve_ivp.solve_ivp_phase import SolveIVPPhase
from dymos.phases.runge_kutta.test.rk_test_ode import TestODE, _test_ode_solution
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestSolveIVPSimpleIntegration(unittest.TestCase):

    def test_simple_integration_forward(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.initial_states:y'] = 0.5

        p.run_model()

        expected = _test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])
        assert_rel_error(self, p['phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_simple_integration_backward(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 2.0
        p['phase0.t_duration'] = -2.0

        p['phase0.initial_states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])).T
        assert_rel_error(self, p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)

    def test_simple_integration_forward_dense(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=TestODE,
                                                              output_nodes_per_seg=20))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.initial_states:y'] = 0.5

        p.run_model()

        expected = _test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])
        assert_rel_error(self, p['phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_simple_integration_backward_dense(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=TestODE,
                                                              output_nodes_per_seg=20))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 2.0
        p['phase0.t_duration'] = -2.0

        p['phase0.initial_states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])).T
        assert_rel_error(self, p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)

class TestSolveIVPWithControls(unittest.TestCase):

    def test_solve_ivp_brachistochrone_solution(self):
        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=BrachistochroneODE))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_input_parameter('g', units='m/s**2', val=9.80665)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.8016

        p['phase0.initial_states:x'] = 0.0
        p['phase0.initial_states:y'] = 10.0
        p['phase0.initial_states:v'] = 0.0

        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.input_parameters:g'] = 9.80665

        p.run_model()

    def test_simple_integration_backward(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=TestODE))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 2.0
        p['phase0.t_duration'] = -2.0

        p['phase0.initial_states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])).T
        assert_rel_error(self, p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)


if __name__ == '__main__':
    unittest.main()
