from .grid_refinement.ph_adaptive.ph_adaptive import PHAdaptive
from .phase.phase import Phase

import openmdao.api as om
import dymos as dm
import numpy as np
from dymos.trajectory.trajectory import Trajectory
from dymos.load_case import load_case, find_phases
import os
import sys


def modify_problem(problem, restart=None, reset_grid=False):
    """
    Modifies the problem object by loading in a guess from a specified restart file.

    Parameters
    ----------
    problem : om.Problem
        The problem instance being modified.
    restart : String or None
        The name of a database to use for restarting the problem.
    reset_grid: Boolean
        Flag to trigger a grid reset.
    """
    # record variables to database when running driver under hook
    # pre-hook is important, because recording initialization is skipped if final_setup has run once
    save_db = os.getcwd() + '/dymos_solution.db'

    try:
        os.remove(save_db)
    except FileNotFoundError:
        pass  # OK if old database is not present to be deleted

    print('adding recorder at:', save_db)
    problem.driver.add_recorder(om.SqliteRecorder(save_db))
    problem.driver.recording_options['includes'] = ['*']
    problem.driver.recording_options['record_inputs'] = True
    # problem.record_iteration('final')    # TODO: not working to save only last iteration?

    # if opts.get('reset_grid'):  # TODO: implement this option
    #     pass

    if restart is not None:  # restore variables from database file specified by 'restart'
        print('Restarting run_problem using the %s database.' % restart)
        cr = om.CaseReader(restart)
        cases = cr.list_cases()
        if len(cases) < 1:
            print('WARNING: the requested %s database file does not have any cases to load.' % restart)
        else:
            case = cr.get_case(cases[-1])  # TODO: use last case, ideally it should be the only one, but there are many

            # identify database as a simulation output and not a solution recording, based on the content
            check_simulation = cr.problem_metadata['driver']['name'] == 'Driver'
            if check_simulation:
                prev_soln = {'inputs':  case.list_inputs(out_stream=None,  units=True, prom_name=True),
                             'outputs': case.list_outputs(out_stream=None, units=True, prom_name=True)}

                load_case(problem, prev_soln)
            else:
                # Initialize the system with values from the case.
                # We unnecessarily call setup again just to make sure we obliterate the previous solution
                # First reset the connections at the top level model until fixed in OpenMDAO
                problem.setup()

                # Load the values from the previous solution
                load_case(problem, case)


def run_problem(problem, refine=False, refine_iteration_limit=10, run_driver=True, simulate=False, no_iterate=False):
    """
    A Dymos-specific interface to execute an OpenMDAO problem containing Dymos Trajectories or
    Phases.  This function can iteratively call run_driver to perform grid refinement, and automatically
    call simulate following a run to check the validity of a result.

    Parameters
    ----------
    problem : om.Problem
        The OpenMDAO problem object to be run.
    refine : bool
        If True, perform grid refinement on the Phases found in the Problem.
    refine_iteration_limit : int
        The number of passes through the grid refinement algorithm to be made.
    run_driver : bool
        If True, run the driver (optimize the problem), otherwise just run the model one time.
    no_iterate : bool
        If True, run the driver but do not iterate.
    simulate : bool
        If True, perform a simulation of Trajectories found in the Problem after the driver
        has been run and grid refinement is complete.
    """
    problem.final_setup()  # make sure command line option hook has a chance to run

    if run_driver:
        if no_iterate:
            problem.driver.opt_settings['maxiter'] = 0
        problem.run_driver()
    else:
        problem.run_model()

    if refine and refine_iteration_limit > 0 and run_driver:
        out_file = 'grid_refinement.out'

        phases = find_phases(problem.model)

        ref = PHAdaptive(phases)
        with open(out_file, 'w+') as f:

            for i in range(refine_iteration_limit):
                refine_results = ref.check_error()

                ref.refine(refine_results)

                for stream in f, sys.stdout:
                    ref.write_iteration(stream, i, phases, refine_results)

                refined_phases = [phase_path for phase_path in refine_results if
                                  phases[phase_path].refine_options['refine'] and
                                  np.any(refine_results[phase_path]['need_refinement'])]

                if not refined_phases:
                    break

                prev_soln = {'inputs': problem.model.list_inputs(out_stream=None, units=True, prom_name=True),
                             'outputs': problem.model.list_outputs(out_stream=None, units=True, prom_name=True)}

                # TODO: Until this is fixed in OpenMDAO 3.0.1
                if isinstance(problem.driver, om.pyOptSparseDriver):
                    problem.driver._res_jacs = {}

                problem.setup()

                load_case(problem, prev_soln)

                problem.run_driver()
            for stream in [f, sys.stdout]:
                if i == refine_iteration_limit-1:
                    print('Iteration limit exceeded. Unable to satisfy specified tolerance', file=stream)
                else:
                    print('Successfully completed grid refinement.', file=stream)
            print(50 * '=')

    if simulate:
        for subsys, local in problem.model._all_subsystem_iter():
            if isinstance(subsys, Trajectory):
                subsys.simulate(record_file='dymos_simulation.db')
