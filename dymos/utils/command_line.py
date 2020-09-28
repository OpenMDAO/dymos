import openmdao.utils.hooks as hooks
import argparse
import sys
import os
from dymos.run_problem import modify_problem, run_problem
from unittest.mock import patch


def _simple_exec(script_name, user_args):
    """
    Use this as executor for commands that run as Problem commands.

    Parameters
    ----------
    script_name : string
        Name of the script to run.
    user_args: list of strings
        Any user supplied arguments for the script.
    """
    # use patch to avoid writing to sys.argv and sys.path
    with patch.object(sys, 'argv', [script_name] + user_args):
        with patch.object(sys, 'path', [os.path.dirname(script_name)] + sys.path[1:]):

            with open(script_name, 'rb') as fp:
                code = compile(fp.read(), script_name, 'exec')

            globals_dict = {
                '__file__': script_name,
                '__name__': '__main__',
                '__package__': None,
                '__cached__': None,
            }

            exec(code, globals_dict)

    return globals_dict


def dymos_cmd(argv=None):
    # pre-parse sys.argv to split between before and after '--'
    alt_args = argv if argv else sys.argv
    if '--' in alt_args:
        idx = alt_args.index('--')
        sys_args = alt_args[:idx]
        user_args = alt_args[idx + 1:]
        argv = sys_args  # don't change sys.argv
    else:
        user_args = []

    parser = argparse.ArgumentParser(description='Dymos Command Line Tool')
    parser.add_argument('script', type=str,
                        help='Python script that creates a Dymos problem to run')
    parser.add_argument('-s', '--simulate', action='store_true',
                        help='If given, perform simulation after solving the problem.')
    parser.add_argument('-n', '--no_solve', action='store_true',
                        help='If given, do not run the driver on the problem (simulate only)')
    parser.add_argument('-t', '--solution', default=None,
                        help='A previous solution record file (or explicit simulation record file)'
                             ' from which the initial guess should be loaded. (default: None)')
    parser.add_argument('-r', '--reset_grid', action='store_true',
                        help='If given, reset the grid to the specs given in the problem definition'
                             ' instead of the grid associated with the loaded solution.')
    parser.add_argument('-l', '--refine_limit', default=0,
                        help='The number of passes through the grid refinement algorithm'
                             ' to use. (default: 0)')
    args = parser.parse_args(argv)  # sys.argv is used if argv parameter is None

    if args.solution == 'dymos_solution.db':  # make sure the loaded db is not being overwritten by the new db
        db_copy = 'old_dymos_solution.db'
        os.rename('dymos_solution.db', db_copy)
        args.solution = db_copy

    opts = {
        'refine_iteration_limit': int(args.refine_limit),
        'restart': args.solution,
        'simulate': args.simulate,
        'no_solve': args.no_solve,
        'reset_grid': args.reset_grid
    }

    class DymosHooks:
        """
        DymosHooks is a lightweight class which provides the Dymos command-line OpenMDAO
        hooks with a state.

        Attributes
        ----------
        _hooks_enabled : bool
            When True, the associated hooks will be allowed to run.
        """

        def __init__(self):
            self._pre_hooks_enabled = True
            self._post_hooks_enabled = False

        def _pre_final_setup(self, prob):
            if not self._pre_hooks_enabled:
                return
            self._pre_hooks_enabled = False

            modify_problem(prob, restart=opts['restart'])
            self._post_hooks_enabled = True

        def _post_final_setup(self, prob):
            if not self._post_hooks_enabled:
                return
            self._post_hooks_enabled = False

            refine_iterations = opts.get('refine_iteration_limit')
            run_problem(prob, refine_iteration_limit=refine_iterations,
                        run_driver=not opts['no_solve'], simulate=opts['simulate'])

    dymos_hooks = DymosHooks()

    hooks.use_hooks = True
    hooks._register_hook('final_setup',
                         'Problem',
                         pre=dymos_hooks._pre_final_setup,
                         post=dymos_hooks._post_final_setup)

    globals_dict = _simple_exec(args.script, user_args)  # run the script

    if not sys.argv[0].endswith('dymos'):  # suppress printing return value when running from command line
        return globals_dict  # return globals for possible checking in unit tests


if __name__ == '__main__':
    dymos_cmd()
