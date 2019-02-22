from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, NonlinearBlockGS, NonlinearRunOnce, ScipyOptimizeDriver, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import RungeKuttaPhase
from dymos.phases.runge_kutta.test.rk_test_ode import TestODE, test_ode_solution
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestRK4WithControls(unittest.TestCase):

    def test_brachistochrone_forward_fixed_initial(self):

        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3

        # p.driver.options['dynamic_simul_derivs'] = True

        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=10,
                            method='rk4',
                            ode_class=BrachistochroneODE,
                            k_solver_options={'iprint': 2},
                            continuity_solver_options={'iprint': 2, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2

        p['phase0.states:x'] = 0 #  phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = 10 #  phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = 0 # phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[1, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_model()

        c = p.check_partials(method='fd', compact_print=True)
        p.run_driver()

        # print(p['phase0.timeseries.time'], p['phase0.timeseries.design_parameters:g'])
        # print(p['phase0.timeseries.time'], p['phase0.timeseries.controls:theta'])
        import matplotlib.pyplot as plt
        plt.plot(p.get_val('phase0.timeseries.states:x'), p.get_val('phase0.timeseries.states:y'))
        plt.show()

        # from openmdao.api import view_model
        # view_model(p.model)

        #expected = test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])
        #assert_rel_error(self, p['phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_single_segment_simple_integration_backward_fixed_final(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=200,
                            method='rk4',
                            ode_class=TestODE,
                            direction='backward',
                            k_solver_options={'iprint': 2},
                            continuity_solver_options={'iprint': 2, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=False, fix_final=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])).T
        assert_rel_error(self, p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)

    def test_single_segment_simple_integration_forward_fixed_final(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=200,
                            method='rk4',
                            ode_class=TestODE,
                            k_solver_options={'iprint': 2},
                            continuity_solver_options={'iprint': 2, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, fix_duration=True)

        expected = 'When RungeKuttaPhase option \'direction\' is \'forward\', state ' \
                   'option \'fix_initial\' must be True.'
        with self.assertRaises(ValueError) as e:
            phase.set_state_options('y', fix_final=True)
        self.assertEqual(str(e.exception), expected)

    def test_single_segment_simple_integration_backward_fixed_initial(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=200,
                            method='rk4',
                            ode_class=TestODE,
                            direction='backward',
                            k_solver_options={'iprint': 2},
                            continuity_solver_options={'iprint': 2, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, fix_duration=True)

        expected = 'When RungeKuttaPhase option \'direction\' is \'backward\', state ' \
                   'option \'fix_final\' must be True.'
        with self.assertRaises(ValueError) as e:
            phase.set_state_options('y', fix_initial=True)
        self.assertEqual(str(e.exception), expected)

if __name__ == '__main__':
    unittest.main()