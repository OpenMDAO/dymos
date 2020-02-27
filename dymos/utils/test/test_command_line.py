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
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_args = ['dymos_testing', os.path.join(test_dir, 'brachistochrone_for_command_line.py')]

        print('Removing the stale test databases before running.')
        for filename in ['dymos_solution.db', 'old_dymos_solution.db', 'grid_refinement.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def _assert_correct_solution(self):
        # Fail if the recorded driver solution file does not exist (driver did not execute)
        self.assertTrue(os.path.exists('dymos_solution.db'))

        # Assert the results are what we expect.
        cr = om.CaseReader('dymos_solution.db')

        # Make sure there are cases
        num_cases = len(cr.list_cases())
        self.assertTrue(num_cases > 0)

        # If there are cases, get the last one
        case = cr.get_case(-1)

        # Make sure the driver converged
        self.assertTrue(case.success)

    # def test_ex_brachistochrone_stock(self):
    #     """ Test to verify that the command line interface intercepts final_setup and runs
    #     dm.run_problem by default without any additional arguments. """
    #     print('test_ex_brachistochrone_stock')
    #     with patch.object(sys, 'argv', self.base_args):
    #         globals_dict = command_line.dymos_cmd()
    #
    #     self._assert_correct_solution()
    #     # check first part of controls result:
    #     assert_almost_equal(globals_dict['p']['traj0.phase0.controls:theta'][:3],
    #                         np.array([[2.44713004], [1.08682697], [1.32429102]]))

    # TODO: causes random seeming testflo errors:
    def test_ex_brachistochrone_stock_nosolve_nosim(self):
        """ Test to verify that the command line interface intercepts final_setup and
        does nothing if given `--no_solve` and not given `--simulate`. """
        print('test_ex_brachistochrone_stock_nosolve_nosim')
        with patch.object(sys, 'argv', self.base_args + ['--no_solve']):
            command_line.dymos_cmd()

        self.assertTrue(os.path.exists('dymos_solution.db'))
        cr = om.CaseReader('dymos_solution.db')
        self.assertTrue(len(cr.list_cases()) == 0)  # no case recorded

    # TODO: causes random seeming testflo errors:
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
        self.assertTrue(os.path.exists('old_dymos_solution.db'))  # old database renamed when used as input

    # TODO: causes random seeming testflo errors:
    def test_ex_brachistochrone_no_solve(self):
        print('test_ex_brachistochrone_no_solve')
        with patch.object(sys, 'argv', self.base_args + ['--no_solve']):
            command_line.dymos_cmd()

        self.assertTrue(os.path.exists('dymos_solution.db'))
        cr = om.CaseReader('dymos_solution.db')
        self.assertTrue(len(cr.list_cases()) == 0)  # no case recorded

    def test_ex_brachistochrone_simulate(self):
        print('test_ex_brachistochrone_simulate')
        with patch.object(sys, 'argv', self.base_args + ['--simulate']):
            command_line.dymos_cmd()

        self._assert_correct_solution()

    def test_ex_brachistochrone_reset_grid(self):
        print('test_ex_brachistochrone_reset_grid')
        with patch.object(sys, 'argv', self.base_args + ['--reset_grid']):
            command_line.dymos_cmd()

        self._assert_correct_solution()


if __name__ == '__main__':
    unittest.main()
