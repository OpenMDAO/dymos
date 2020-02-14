import unittest
import dymos.utils.command_line as command_line
from openmdao.utils.testing_utils import use_tempdirs
import sys
import os


# TODO: these tests currently just check that the problem runs, they should check results
@use_tempdirs
class TestCommandLine(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))

        print('Removing the stale test databases before running.')
        for db in ['dymos_solution.db', 'old_dymos_solution.db']:
            try:
                os.remove(db)
            except FileNotFoundError:
                pass  # OK if old database is not present to be deleted

    def test_ex_brachistochrone_stock(self):
        print('test_ex_brachistochrone_stock')
        sys.argv = ['dymos',
                    self.test_dir  + '/../../examples/brachistochrone/test/ex_brachistochrone.py']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_iteration(self):
        print('test_ex_brachistochrone_iteration')
        sys.argv = ['dymos',
                    self.test_dir  + '/../../examples/brachistochrone/test/ex_brachistochrone.py',
                    '--refine_limit=5']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_solution(self):
        # run stock problem first to record the output database
        print('test_ex_brachistochrone_solution first run')
        sys.argv = ['dymos',
                    self.test_dir  + '/../../examples/brachistochrone/test/ex_brachistochrone.py']
        command_line.dymos_cmd()

        # run problem again loading the output database
        print('test_ex_brachistochrone_solution second run')
        sys.argv = ['dymos',
                    self.test_dir  + '/../../examples/brachistochrone/test/ex_brachistochrone.py',
                    '--solution=dymos_solution.db']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_no_solve(self):
        print('test_ex_brachistochrone_no_solve')
        sys.argv = ['dymos',
                    self.test_dir  + '/../../examples/brachistochrone/test/ex_brachistochrone.py',
                    '--no_solve']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_simulate(self):
        print('test_ex_brachistochrone_simulate')
        sys.argv = ['dymos',
                    self.test_dir  + '/../../examples/brachistochrone/test/ex_brachistochrone.py',
                    '--simulate']
        command_line.dymos_cmd()

    def test_ex_brachistochrone_reset_grid(self):
        print('test_ex_brachistochrone_reset_grid')
        sys.argv = ['dymos',
                    self.test_dir  + '/../../examples/brachistochrone/test/ex_brachistochrone.py',
                    '--reset_grid']
        command_line.dymos_cmd()


if __name__ == '__main__':
    unittest.main()
