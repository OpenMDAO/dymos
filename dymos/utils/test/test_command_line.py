import unittest
import dymos.utils.command_line as command_line
from openmdao.utils.testing_utils import use_tempdirs
import sys
import os
import openmdao.api as om


@use_tempdirs
class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        print('Removing the stale test databases before running.')
        for db in ['dymos_solution.db', 'old_dymos_solution.db', 'grid_refinement.out']:
            try:
                os.remove(db)
            except FileNotFoundError:
                pass  # OK if old database is not present to be deleted

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

    def test_ex_brachistochrone_stock(self):
        """ Test to verify that the command line interface intercepts final_setup and runs
        dm.run_problem by default without any additional arguments. """
        print('test_ex_brachistochrone_stock')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py')]
        command_line.dymos_cmd()
        self._assert_correct_solution()

    def test_ex_brachistochrone_stock_nosolve_nosim(self):
        """ Test to verify that the command line interface intercepts final_setup and
        does nothing if given `--no_solve` and not given `--simulate`. """
        print('test_ex_brachistochrone_stock')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py'),
                    '--no_solve']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_iteration(self):
        print('test_ex_brachistochrone_iteration')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py'),
                    '--refine_limit=5']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_solution(self):
        # run stock problem first to record the output database
        print('test_ex_brachistochrone_solution first run')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py')]
        command_line.dymos_cmd()

        # run problem again loading the output database
        print('test_ex_brachistochrone_solution second run')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py'),
                    '--solution=dymos_solution.db']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_no_solve(self):
        print('test_ex_brachistochrone_no_solve')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py'),
                    '--no_solve']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_simulate(self):
        print('test_ex_brachistochrone_simulate')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py'),
                    '--simulate']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_reset_grid(self):
        print('test_ex_brachistochrone_reset_grid')
        sys.argv = ['dymos',
                    os.path.join(self.test_dir, 'brachistochrone_for_command_line.py'),
                    '--reset_grid']
        command_line.dymos_cmd()


if __name__ == '__main__':
    unittest.main()
