from __future__ import print_function, absolute_import, division

import os
import unittest

import numpy as np

from openmdao.utils.assert_utils import assert_rel_error


class TestPhaseSimulationResults(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase_simulation_test_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use('Agg')
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
        from dymos.utils.simulation.phase_simulation_results import PhaseSimulationResults

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='rad', rate_continuity=False, lower=0.001, upper=3.14)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

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

        cls.exp_out = phase.simulate(times=100, record_file='phase_simulation_test_sim.db')

        cls.exp_out_loaded = PhaseSimulationResults('phase_simulation_test_sim.db')

    def test_returned_and_loaded_equivalent(self):
        assert_rel_error(self,
                         self.exp_out.outputs['indep']['time']['value'],
                         self.exp_out_loaded.outputs['indep']['time']['value'])

        for var_type in ('states', 'controls', 'control_rates', 'design_parameters'):
            exp_var_keys = set(self.exp_out.outputs[var_type].keys())
            loaded_exp_var_keys = set(self.exp_out_loaded.outputs[var_type].keys())

            self.assertEqual(exp_var_keys, loaded_exp_var_keys)

            for var in exp_var_keys:
                self.assertEqual(self.exp_out.outputs[var_type][var]['units'],
                                 self.exp_out_loaded.outputs[var_type][var]['units'])

                constructed_shape = self.exp_out.outputs[var_type][var]['shape']
                loaded_shape = self.exp_out_loaded.outputs[var_type][var]['shape']
                self.assertEqual(constructed_shape,
                                 loaded_shape,
                                 msg='different shapes returned PhaseSimulationResults vs '
                                     'loaded PhaseSimulationResults: {0} - loaded shape: {1} - '
                                     'returned shape: {2}'.format(var, loaded_shape,
                                                                  constructed_shape))
                assert_rel_error(self,
                                 self.exp_out.outputs[var_type][var]['value'],
                                 self.exp_out_loaded.outputs[var_type][var]['value'])

    def test_get_values_equivalent(self):

        for var in ('time', 'x', 'y', 'v', 'theta', 'theta_rate', 'theta_rate2', 'check'):
            assert_rel_error(self,
                             self.exp_out.get_values(var),
                             self.exp_out_loaded.get_values(var))

    def test_load_invalid_var(self):

        for source in self.exp_out, self.exp_out_loaded:
            with self.assertRaises(ValueError) as ctx:
                source.get_values('foo')
            self.assertEqual(str(ctx.exception),
                             'Variable "foo" not found in phase simulation results.')

    def test_convert_units(self):

        units = {'time': 'min',
                 'x': 'ft',
                 'y': 'ft',
                 'v': 'ft/s',
                 'g': 'ft/s**2',
                 'theta': 'deg',
                 'theta_rate': 'deg/s',
                 'theta_rate2': 'deg/s**2',
                 'check': 'ft/min'}

        conv = {'time': 1/60.,
                'x': 3.2808399,
                'y': 3.2808399,
                'v': 3.2808399,
                'g': 3.2808399,
                'theta': 180.0/np.pi,
                'theta_rate': 180.0/np.pi,
                'theta_rate2': 180.0/np.pi,
                'check': 196.85039370079}

        for var in ('time', 'x', 'y', 'v', 'g', 'theta', 'theta_rate', 'theta_rate2', 'check'):
            assert_rel_error(self,
                             self.exp_out.get_values(var, units=units[var]),
                             self.exp_out.get_values(var) * conv[var],
                             tolerance=1.0E-8)

            assert_rel_error(self,
                             self.exp_out_loaded.get_values(var, units=units[var]),
                             self.exp_out_loaded.get_values(var) * conv[var],
                             tolerance=1.0E-8)
