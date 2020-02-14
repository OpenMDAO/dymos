import openmdao.utils.hooks as hooks
import argparse
import sys
import os
from dymos.run_problem import modify_problem
import dymos.utils.command_line as cl

modify_enabled = True


def _simple_exec(script_name, pre_hook_function, user_args):
    """
    Use this as executor for commands that run as Problem commands.

    Parameters
    ----------
    script_name : string
        Name of the script to run.
    pre_hook_function: function
        final_setup hook function.
    user_args: list of strings
        Any user supplied arguments for the script.
    """

    sys.path.insert(0, os.path.dirname(script_name))
    sys.argv[:] = [script_name] + user_args

    with open(script_name, 'rb') as fp:
        code = compile(fp.read(), script_name, 'exec')

    globals_dict = {
        '__file__': script_name,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    pre_hook_function(0)  # set up the hook function
    exec(code, globals_dict)


def dymos_cmd():
    # pre-parse sys.argv to split between before and after '--'
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        sys_args = sys.argv[:idx]
        user_args = sys.argv[idx + 1:]
        sys.argv[:] = sys_args
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
    args = parser.parse_args()

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

    hooks.use_hooks = True
    cl.modify_enabled = True  # enable hook's effect

    def _pre_final_setup(prob):
        if not cl.modify_enabled:  # unregistering the hook does not allow it to be reliably re-enabled
            return

        modify_problem(prob, opts)
        cl.modify_enabled = False  # disable hook's effect

    hooks._register_hook('final_setup', 'Problem', pre=_pre_final_setup)  # enable pre-hook
    _simple_exec(args.script, lambda _: _pre_final_setup, user_args)


if __name__ == '__main__':
    dymos_cmd()
