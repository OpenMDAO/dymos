from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from scipy.interpolate import interp1d

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
        p['phase0.t_duration'] = 1.8016043

        p['phase0.initial_states:x'] = 0.0
        p['phase0.initial_states:y'] = 10.0
        p['phase0.initial_states:v'] = 0.0

        p['phase0.controls:theta'] = phase.interpolate(ys=[0.01, 1.00501645e+02],
                                                       nodes='control_input')
        p['phase0.input_parameters:g'] = 9.80665

        p.run_model()

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10.0, tolerance=1.0E-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5.0, tolerance=1.0E-4)

    def test_solve_ivp_brachistochrone_solution_dense(self):
        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=BrachistochroneODE,
                                                              output_nodes_per_seg=20))

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

        p['phase0.controls:theta'] = phase.interpolate(ys=[0.01, 1.00501645e+02],
                                                       nodes='control_input')
        p['phase0.input_parameters:g'] = 9.80665

        p.run_model()

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10.0, tolerance=1.0E-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5.0, tolerance=1.0E-4)

    def test_solve_ivp_brachistochrone_solution_design_param(self):
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

        phase.add_design_parameter('g', units='m/s**2', val=1.0)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.8016043

        p['phase0.initial_states:x'] = 0.0
        p['phase0.initial_states:y'] = 10.0
        p['phase0.initial_states:v'] = 0.0

        p['phase0.controls:theta'] = phase.interpolate(ys=[0.01, 1.00501645e+02],
                                                       nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_model()

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10.0, tolerance=1.0E-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5.0, tolerance=1.0E-4)

    def test_solve_ivp_brachistochrone_solution_dense_design_param(self):
        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=BrachistochroneODE,
                                                              output_nodes_per_seg=20))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', val=1.0)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.8016

        p['phase0.initial_states:x'] = 0.0
        p['phase0.initial_states:y'] = 10.0
        p['phase0.initial_states:v'] = 0.0

        p['phase0.controls:theta'] = phase.interpolate(ys=[0.01, 1.00501645e+02],
                                                       nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_model()

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10.0, tolerance=1.0E-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5.0, tolerance=1.0E-4)


class TestSolveIVPWithPolynomialControls(unittest.TestCase):

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

        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

        phase.add_input_parameter('g', units='m/s**2', val=9.80665)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.8016043

        p['phase0.initial_states:x'] = 0.0
        p['phase0.initial_states:y'] = 10.0
        p['phase0.initial_states:v'] = 0.0

        p['phase0.polynomial_controls:theta'] = [[0.01], [1.00501645e+02]]
        p['phase0.input_parameters:g'] = 9.80665

        p.run_model()

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10.0, tolerance=1.0E-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5.0, tolerance=1.0E-4)

    def test_solve_ivp_brachistochrone_solution_dense(self):
        p = Problem(model=Group())
        phase = p.model.add_subsystem('phase0', SolveIVPPhase(num_segments=4,
                                                              method='RK45',
                                                              atol=1.0E-12,
                                                              rtol=1.0E-12,
                                                              ode_class=BrachistochroneODE,
                                                              output_nodes_per_seg=20))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

        phase.add_input_parameter('g', units='m/s**2', val=9.80665)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.8016043

        p['phase0.initial_states:x'] = 0.0
        p['phase0.initial_states:y'] = 10.0
        p['phase0.initial_states:v'] = 0.0

        p['phase0.polynomial_controls:theta'] = [[0.01], [1.00501645e+02]]
        p['phase0.input_parameters:g'] = 9.80665

        p.run_model()

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10.0, tolerance=1.0E-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5.0, tolerance=1.0E-4)


class TestSolveIVPPhaseCopy(unittest.TestCase):

    def test_copy_brachistochrone(self):
        from openmdao.api import ScipyOptimizeDriver, DirectSolver
        from dymos import DeprecatedPhaseFactory

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = DeprecatedPhaseFactory('gauss-lobatto',
                                       ode_class=BrachistochroneODE,
                                       transcription_order=3,
                                       num_segments=20)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

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

        sim_prob = Problem(model=Group())
        sim_phase = SolveIVPPhase(from_phase=phase,
                                  atol=1.0E-12,
                                  rtol=1.0E-12,
                                  output_nodes_per_seg=20)

        sim_prob.model.add_subsystem(phase.name,
                                     subsys=sim_phase)

        sim_prob.setup()

        sim_prob.set_val('phase0.t_initial', p.get_val('phase0.t_initial'))
        sim_prob.set_val('phase0.t_duration', p.get_val('phase0.t_duration'))

        sim_prob.set_val('phase0.initial_states:x', p.get_val('phase0.states:x')[0, ...])
        sim_prob.set_val('phase0.initial_states:y', p.get_val('phase0.states:y')[0, ...])
        sim_prob.set_val('phase0.initial_states:v', p.get_val('phase0.states:v')[0, ...])

        sim_prob.set_val('phase0.controls:theta', p.get_val('phase0.controls:theta'))

        sim_prob.set_val('phase0.design_parameters:g', p.get_val('phase0.design_parameters:g'))

        sim_prob.run_model()

        x_sol = p.get_val('phase0.timeseries.states:x')
        y_sol = p.get_val('phase0.timeseries.states:y')
        v_sol = p.get_val('phase0.timeseries.states:v')
        theta_sol = p.get_val('phase0.timeseries.controls:theta')
        time_sol = p.get_val('phase0.timeseries.time')

        x_sim = sim_prob.get_val('phase0.timeseries.states:x')
        y_sim = sim_prob.get_val('phase0.timeseries.states:y')
        v_sim = sim_prob.get_val('phase0.timeseries.states:v')
        theta_sim = sim_prob.get_val('phase0.timeseries.controls:theta')
        time_sim = sim_prob.get_val('phase0.timeseries.time')

        x_interp = interp1d(time_sim[:, 0], x_sim[:, 0])
        y_interp = interp1d(time_sim[:, 0], y_sim[:, 0])
        v_interp = interp1d(time_sim[:, 0], v_sim[:, 0])
        theta_interp = interp1d(time_sim[:, 0], theta_sim[:, 0])

        assert_rel_error(self, x_interp(time_sol), x_sol, tolerance=1.0E-5)
        assert_rel_error(self, y_interp(time_sol), y_sol, tolerance=1.0E-5)
        assert_rel_error(self, v_interp(time_sol), v_sol, tolerance=1.0E-5)
        assert_rel_error(self, theta_interp(time_sol), theta_sol, tolerance=1.0E-5)


if __name__ == '__main__':
    unittest.main()
