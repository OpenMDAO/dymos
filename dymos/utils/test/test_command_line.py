import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_almost_equal
import dymos.utils.command_line as command_line
from openmdao.utils.testing_utils import use_tempdirs
import sys
import os
import openmdao.api as om


@use_tempdirs
class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_args = ['dymos_testing', os.path.join(self.test_dir,
                                                        'brachistochrone_for_command_line.py')]

    def _assert_correct_solution(self):
        # Fail if the recorded driver solution file does not exist (driver did not execute)
        self.assertTrue(os.path.exists('dymos_solution.db'))

        # Assert the results are what we expect.
        cr = om.CaseReader('dymos_solution.db')
        case = cr.get_case('final')

        # Make sure the driver converged
        self.assertTrue(case.success)

    def test_ex_brachistochrone_stock(self):
        """ Test to verify that the command line interface intercepts final_setup and runs
        dm.run_problem by default without any additional arguments. """
        print('test_ex_brachistochrone_stock')
        with patch.object(sys, 'argv', self.base_args):
            globals_dict = command_line.dymos_cmd()

        self._assert_correct_solution()
        print(globals_dict['p']['traj0.phase0.controls:theta'][:3])
        # check first part of controls result:
        assert_almost_equal(globals_dict['p']['traj0.phase0.controls:theta'][:3],
                            np.array([[2.54206362], [4.8278643], [10.11278149]]))

    def test_ex_brachistochrone_stock_nosolve_nosim(self):
        """ Test to verify that the command line interface intercepts final_setup and
        does nothing if given `--no_solve` and not given `--simulate`. """
        print('test_ex_brachistochrone_stock_nosolve_nosim')
        with patch.object(sys, 'argv', self.base_args + ['--no_solve']):
            command_line.dymos_cmd()

        self.assertTrue(os.path.exists('dymos_solution.db'))
        cr = om.CaseReader('dymos_solution.db')
        self.assertTrue(len(cr.list_cases()) == 1)

    def test_ex_brachistochrone_iteration(self):
        print('test_ex_brachistochrone_iteration')
        with patch.object(sys, 'argv', self.base_args + ['--refine_limit=5']):
            command_line.dymos_cmd()

        self._assert_correct_solution()
        self.assertTrue(os.path.exists('grid_refinement.out'))

    def test_ex_brachistochrone_solution(self):
        # run stock problem first to record the output database
        print('test_ex_brachistochrone_solution first run')
        with patch.object(sys, 'argv', self.base_args):
            command_line.dymos_cmd()

        # run problem again loading the output database
        print('test_ex_brachistochrone_solution second run')
        with patch.object(sys, 'argv', self.base_args + ['--solution=dymos_solution.db']):
            command_line.dymos_cmd()

        self._assert_correct_solution()
        # old database renamed when used as input
        self.assertTrue(os.path.exists('old_dymos_solution.db'))

    def test_ex_brachistochrone_no_solve(self):
        print('test_ex_brachistochrone_no_solve')
        with patch.object(sys, 'argv', self.base_args + ['--no_solve']):
            command_line.dymos_cmd()

        self.assertTrue(os.path.exists('dymos_solution.db'))
        cr = om.CaseReader('dymos_solution.db')
        self.assertTrue(len(cr.list_cases()) == 1)

    def test_ex_brachistochrone_simulate(self):
        print('test_ex_brachistochrone_simulate')
        with patch.object(sys, 'argv', self.base_args + ['--simulate']):
            command_line.dymos_cmd()

        self._assert_correct_solution()
        self.assertTrue(os.path.exists('dymos_solution.db'))
        self.assertTrue(os.path.exists('dymos_simulation.db'))

        cr = om.CaseReader('dymos_simulation.db')
        self.assertListEqual(['final'], cr.list_cases())
        case = cr.get_case('final')
        self.assertEqual(57, len(case.outputs))

    @unittest.skipIf(True, reason='grid resetting not yet implemented')
    def test_ex_brachistochrone_reset_grid(self):
        print('test_ex_brachistochrone_reset_grid')
        with patch.object(sys, 'argv', self.base_args + ['--reset_grid']):
            command_line.dymos_cmd()

        self._assert_correct_solution()

    def test_vanderpol_simulation_restart(self):
        from scipy.interpolate import interp1d
        from numpy.testing import assert_almost_equal
        from dymos.examples.vanderpol.vanderpol_dymos_plots import vanderpol_dymos_plots

        self.base_args = ['dymos_testing',
                          os.path.join(self.test_dir,
                                       '../../examples/vanderpol/vanderpol_dymos.py')]

        # run simulation first to record the output database
        print('test_vanderpol_simulation_restart first run')
        with patch.object(sys, 'argv', self.base_args + ['--simulate'] + ['--no_solve']):
            s = command_line.dymos_cmd()

        # run problem again loading the output simulation database, but not solving
        print('test_vanderpol_simulation_restart second run')
        # TODO: need this to match test_modify_problem:test_modify_problem?
        #        q.driver.opt_settings['maxiter'] = 0
        with patch.object(sys, 'argv', self.base_args + ['--solution=dymos_simulation.db']):
            q = command_line.dymos_cmd()

        #  The solution should look like the explicit time history for the states and controls.
        DO_PLOTS = False
        if DO_PLOTS:
            vanderpol_dymos_plots(q['p'])  # only for visual inspection and debug
        else:  # automate comparison
            s['p'] = q['p'].model.traj.simulate()
            # get_val returns data for duplicate time points; remove them before interpolating
            tq = q['p'].get_val('traj.phase0.timeseries.time')[:, 0]
            nodup = np.insert(tq[1:] != tq[:-1], 0, True)
            tq = tq[nodup]
            x1q = q['p'].get_val('traj.phase0.timeseries.states:x1')[:, 0][nodup]
            x0q = q['p'].get_val('traj.phase0.timeseries.states:x0')[:, 0][nodup]
            uq = q['p'].get_val('traj.phase0.timeseries.controls:u')[:, 0][nodup]

            ts = s['p'].get_val('traj.phase0.timeseries.time')[:, 0]
            nodup = np.insert(ts[1:] != ts[:-1], 0, True)
            ts = ts[nodup]
            x1s = s['p'].get_val('traj.phase0.timeseries.states:x1')[:, 0][nodup]
            x0s = s['p'].get_val('traj.phase0.timeseries.states:x0')[:, 0][nodup]
            us = s['p'].get_val('traj.phase0.timeseries.controls:u')[:, 0][nodup]

            # create interpolation functions so that values can be looked up at matching time points
            fx1s = interp1d(ts, x1s, kind='cubic')
            fx0s = interp1d(ts, x0s, kind='cubic')
            fus = interp1d(ts, us, kind='cubic')

            assert_almost_equal(x1q, fx1s(tq), decimal=2)
            assert_almost_equal(x0q, fx0s(tq), decimal=2)
            assert_almost_equal(uq, fus(tq), decimal=5)

    def test_ex_brachistochrone_make_plots(self):
        print('test_ex_brachistochrone_make_plots')
        with patch.object(sys, 'argv', self.base_args + ['--make_plots']):
            command_line.dymos_cmd()

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertTrue(os.path.exists(f'plots/{varname.replace(":","_")}.png'))

    def test_ex_brachistochrone_make_plots_set_plot_dir(self):
        print('test_ex_brachistochrone_make_plots_set_plot_dir')
        plot_dir = 'plot_dir'
        with patch.object(sys, 'argv', self.base_args + ['--make_plots'] +
                          [f'--plot_dir={plot_dir}']):
            command_line.dymos_cmd()

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertTrue(os.path.exists(f'{plot_dir}/{varname.replace(":","_")}.png'))

    def test_ex_brachistochrone_make_no_plots(self):
        print('test_ex_brachistochrone_make_no_plots')
        with patch.object(sys, 'argv', self.base_args):
            command_line.dymos_cmd()

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertFalse(os.path.exists(f'plots/{varname.replace(":","_")}.png'))

    def test_ex_brachistochrone_set_solution_record_file(self):
        print('test_ex_brachistochrone_set_solution_record_file')
        with patch.object(sys, 'argv', self.base_args +
                          ['--solution_record_file=solution_record_file.db']):
            command_line.dymos_cmd()

        self.assertTrue(os.path.exists('solution_record_file.db'))
        cr = om.CaseReader('solution_record_file.db')
        self.assertListEqual(['final'], cr.list_cases())

    def test_ex_brachistochrone_simulate_set_simulation_record_file(self):
        print('test_ex_brachistochrone_simulate')
        with patch.object(sys, 'argv', self.base_args + ['--simulate'] +
                          ['--simulation_record_file=simulation_record_file.db']):
            command_line.dymos_cmd()

        self.assertTrue(os.path.exists('simulation_record_file.db'))
        cr = om.CaseReader('simulation_record_file.db')
        self.assertListEqual(['final'], cr.list_cases())
        case = cr.get_case('final')
        self.assertEqual(57, len(case.outputs))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
