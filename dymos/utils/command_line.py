import openmdao.utils.hooks as hooks
import argparse
import sys
import os
from dymos.run_problem import modify_problem, run_problem
from unittest.mock import patch

hook_options = {
    'pre_hook_enabled': True,
    'post_hook_enabled': True
}


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
            hooks._reset_all_hooks()

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

    hooks.use_hooks = True
    hook_options['pre_hook_enabled'] = True  # enable hook's effect
    hook_options['post_hook_enabled'] = True

    def _pre_final_setup(prob):
        if not hook_options['pre_hook_enabled']:  # unregistering the hook does not allow it to be reliably re-enabled
            return

        hook_options['pre_hook_enabled'] = False  # disable hook's effect

        modify_problem(prob, opts)

    def _post_final_setup(prob):
        if not hook_options['post_hook_enabled']:
            return

        hook_options['post_hook_enabled'] = False  # disable hook's effect

        if not opts['no_solve']:  # execute run_problem unless told otherwise
            refine = opts.get('refine_iteration_limit')
            run_problem(prob, refine, refine_iteration_limit=refine)

    hooks._register_hook('final_setup',
                         'Problem',
                         pre=_pre_final_setup,
                         post=_post_final_setup)

    globals_dict = _simple_exec(args.script, user_args)  # run the script

    if not sys.argv[0].endswith('dymos'):  # suppress printing return value when running from command line
        return globals_dict  # return globals for possible checking in unit tests


if __name__ == '__main__':
    dymos_cmd()
