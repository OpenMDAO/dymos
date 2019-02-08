from __future__ import print_function, absolute_import, division

import os
import unittest

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, CaseReader
from openmdao.utils.assert_utils import assert_rel_error
from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestSimulationPhaseGaussLobatto(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim_gl.sql']:
            if os.path.exists(filename):
                os.remove(filename)

    @classmethod
    def setUpClass(cls):

        p = cls.p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = cls.phase = Phase('gauss-lobatto', ode_class=BrachistochroneODE, num_segments=10,
                                  compressed=False)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate', lower=0, upper=100, units='deg/s')

        phase.add_timeseries_output('check', units='m/s', shape=(1,))

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Setup simulation manually
        cls.sim_prob = phase.simulate(record_file='phase0_sim_gl.sql')

    def test_recorded_simulate_data(self):
        sim_prob = self.sim_prob

        cr = CaseReader('phase0_sim_gl.sql')
        case = cr.get_case(cr.list_cases()[0])

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.time'],
                         sim_prob.get_val('phase0.timeseries.time'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.states:x'],
                         sim_prob.get_val('phase0.timeseries.states:x'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.states:y'],
                         sim_prob.get_val('phase0.timeseries.states:y'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.states:v'],
                         sim_prob.get_val('phase0.timeseries.states:v'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.controls:theta'],
                         sim_prob.get_val('phase0.timeseries.controls:theta'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.check'],
                         sim_prob.get_val('phase0.timeseries.check'))

    def test_simulate_results(self):
        p = self.p

        sim_prob = self.sim_prob

        from scipy.interpolate import interp1d

        t_sol = p.get_val('phase0.timeseries.time')
        x_sol = p.get_val('phase0.timeseries.states:x')
        y_sol = p.get_val('phase0.timeseries.states:y')
        v_sol = p.get_val('phase0.timeseries.states:v')
        theta_sol = p.get_val('phase0.timeseries.controls:theta')

        t_sim = sim_prob.get_val('phase0.timeseries.time')
        x_sim = sim_prob.get_val('phase0.timeseries.states:x')
        y_sim = sim_prob.get_val('phase0.timeseries.states:y')
        v_sim = sim_prob.get_val('phase0.timeseries.states:v')
        theta_sim = sim_prob.get_val('phase0.timeseries.controls:theta')

        f_x = interp1d(t_sim[:, 0], x_sim[:, 0], axis=0)
        f_y = interp1d(t_sim[:, 0], y_sim[:, 0], axis=0)
        f_v = interp1d(t_sim[:, 0], v_sim[:, 0], axis=0)
        f_theta = interp1d(t_sim[:, 0], theta_sim[:, 0], axis=0)

        assert_rel_error(self, f_x(t_sol), x_sol, tolerance=1.0E-3)
        assert_rel_error(self, f_y(t_sol), y_sol, tolerance=1.0E-3)
        assert_rel_error(self, f_v(t_sol), v_sol, tolerance=1.0E-3)
        assert_rel_error(self, f_theta(t_sol), theta_sol, tolerance=1.0E-3)


class TestSimulationPhaseRadau(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim_gl.sql']:
            if os.path.exists(filename):
                os.remove(filename)

    @classmethod
    def setUpClass(cls):

        p = cls.p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = cls.phase = Phase('radau-ps', ode_class=BrachistochroneODE, num_segments=10,
                                  compressed=False)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate', lower=0, upper=100, units='deg/s')

        phase.add_timeseries_output('check', units='m/s', shape=(1,))

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        p.run_model()

        import numpy as np
        np.set_printoptions(linewidth=1024, edgeitems=1000)

        # Solve for the optimal trajectory
        p.run_driver()

        # Setup simulation manually
        cls.sim_prob = phase.simulate(record_file='phase0_sim_gl.sql')

    def test_recorded_simulate_data(self):
        sim_prob = self.sim_prob

        cr = CaseReader('phase0_sim_gl.sql')
        case = cr.get_case(cr.list_cases()[0])

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.time'],
                         sim_prob.get_val('phase0.timeseries.time'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.states:x'],
                         sim_prob.get_val('phase0.timeseries.states:x'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.states:y'],
                         sim_prob.get_val('phase0.timeseries.states:y'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.states:v'],
                         sim_prob.get_val('phase0.timeseries.states:v'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.controls:theta'],
                         sim_prob.get_val('phase0.timeseries.controls:theta'))

        assert_rel_error(self,
                         case.outputs['phase0.timeseries.check'],
                         sim_prob.get_val('phase0.timeseries.check'))

    def test_simulate_results(self):
        p = self.p

        sim_prob = self.sim_prob

        from scipy.interpolate import interp1d

        t_sol = p.get_val('phase0.timeseries.time')
        x_sol = p.get_val('phase0.timeseries.states:x')
        y_sol = p.get_val('phase0.timeseries.states:y')
        v_sol = p.get_val('phase0.timeseries.states:v')
        theta_sol = p.get_val('phase0.timeseries.controls:theta')

        t_sim = sim_prob.get_val('phase0.timeseries.time')
        x_sim = sim_prob.get_val('phase0.timeseries.states:x')
        y_sim = sim_prob.get_val('phase0.timeseries.states:y')
        v_sim = sim_prob.get_val('phase0.timeseries.states:v')
        theta_sim = sim_prob.get_val('phase0.timeseries.controls:theta')

        f_x = interp1d(t_sim[:, 0], x_sim[:, 0], axis=0)
        f_y = interp1d(t_sim[:, 0], y_sim[:, 0], axis=0)
        f_v = interp1d(t_sim[:, 0], v_sim[:, 0], axis=0)
        f_theta = interp1d(t_sim[:, 0], theta_sim[:, 0], axis=0)

        assert_rel_error(self, f_x(t_sol), x_sol, tolerance=1.0E-3)
        assert_rel_error(self, f_y(t_sol), y_sol, tolerance=1.0E-3)
        assert_rel_error(self, f_v(t_sol), v_sol, tolerance=1.0E-3)
        assert_rel_error(self, f_theta(t_sol), theta_sol, tolerance=1.0E-3)

if __name__ == '__main__':
    unittest.main()
