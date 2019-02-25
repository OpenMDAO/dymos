from __future__ import print_function, division, absolute_import

import unittest

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import RungeKuttaPhase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestRK4WithControls(unittest.TestCase):

    def test_brachistochrone_forward_fixed_initial(self):

        p = Problem(model=Group())

        from openmdao.api import pyOptSparseDriver
        p.driver = ScipyOptimizeDriver()
        p.driver = pyOptSparseDriver()

        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=10,
                            method='rk4',
                            ode_class=BrachistochroneODE,
                            k_solver_options={'iprint': -1},
                            continuity_solver_options={'iprint': -1, 'solve_subsystems': True},
                            compressed=True))

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=False,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        p.model.linear_solver = DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[1, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[0], 0, tolerance=1.0E-3)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[0], 10, tolerance=1.0E-3)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:v')[0], 0, tolerance=1.0E-3)

        exp_out = phase.simulate()

        import matplotlib.pyplot as plt
        plt.plot(p.get_val('phase0.timeseries.time'), p.get_val('phase0.timeseries.controls:theta'), 'ro')
        plt.plot(exp_out.get_val('phase0.timeseries.time'), exp_out.get_val('phase0.timeseries.controls:theta'), 'b--')
        plt.show()

    def test_brachistochrone_backward_fixed_final(self):

        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()

        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=10,
                            method='rk4',
                            ode_class=BrachistochroneODE,
                            direction='backward',
                            k_solver_options={'iprint': -1},
                            continuity_solver_options={'iprint': -1, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=False, fix_final=True)
        phase.set_state_options('y', fix_initial=False, fix_final=True)
        phase.set_state_options('v', fix_initial=False, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        phase.add_boundary_constraint('x', loc='initial', equals=0)
        phase.add_boundary_constraint('y', loc='initial', equals=10)
        phase.add_boundary_constraint('v', loc='initial', equals=0)

        p.model.linear_solver = DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2

        p['phase0.states:x'] = 10
        p['phase0.states:y'] = 5
        p['phase0.states:v'] = 10
        p['phase0.controls:theta'] = phase.interpolate(ys=[1, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[0], 0, tolerance=1.0E-3)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[0], 10, tolerance=1.0E-3)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:v')[0], 0, tolerance=1.0E-3)

        import numpy as np
        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials()

    def test_brachistochrone_backward_fixed_final(self):

        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()

        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=10,
                            method='rk4',
                            ode_class=BrachistochroneODE,
                            direction='backward',
                            k_solver_options={'iprint': -1},
                            continuity_solver_options={'iprint': -1, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=False, fix_final=True)
        phase.set_state_options('y', fix_initial=False, fix_final=True)
        phase.set_state_options('v', fix_initial=False, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        phase.add_boundary_constraint('x', loc='initial', equals=0)
        phase.add_boundary_constraint('y', loc='initial', equals=10)
        phase.add_boundary_constraint('v', loc='initial', equals=0)

        p.model.linear_solver = DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2

        p['phase0.states:x'] = 10
        p['phase0.states:y'] = 5
        p['phase0.states:v'] = 10
        p['phase0.controls:theta'] = phase.interpolate(ys=[1, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[0], 0, tolerance=1.0E-3)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:x')[-1], 10, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[0], 10, tolerance=1.0E-3)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 5, tolerance=1.0E-3)

        assert_rel_error(self, p.get_val('phase0.timeseries.states:v')[0], 0, tolerance=1.0E-3)

        import numpy as np
        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials()

if __name__ == '__main__':
    unittest.main()
