from .grid_refinement.ph_adaptive.ph_adaptive import PHAdaptive
from .phase.phase import Phase

import openmdao.api as om
import dymos as dm
import numpy as np
from dymos.trajectory.trajectory import Trajectory
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
    if restart is not None:  # restore variables from database file specified by 'restart'
        print('Restarting run_problem using the %s database.' % restart)
        cr = om.CaseReader(restart)
        cases = cr.list_cases()
        if len(cases) < 1:
            print('WARNING: the requested %s database file does not have any cases to load.')
        else:
            case = cr.get_case(cases[-1])  # TODO: use last case, ideally it should be the only one, but there are many

            # Initialize the system with values from the case.
            # We unnecessarily call setup again just to make sure we obliterate the previous solution
            # First reset the connections at the top level model until fixed in OpenMDAO
            problem.setup()

            # Load the values from the previous solution
            dm.load_case(problem, case)

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


def run_problem(problem, refine=False, refine_iteration_limit=10, run_driver=True, simulate=False):
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
    simulate : bool
        If True, perform a simulation of Trajectories found in the Problem after the driver
        has been run and grid refinement is complete.
    """
    problem.final_setup()  # make sure command line option hook has a chance to run

    if run_driver:
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

                re_interpolate_solution(problem, previous_solution=prev_soln)

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
                subsys.simulate()


def find_phases(sys):
    """
    Finds all instances of Dymos Phases within the given system, and returns them as a dictionary.
    They are keyed by promoted name if use_prom_path=True, otherwise they are keyed by their
    absolute name.

    Parameters
    ----------
    sys : om.Group
        The OpenMDAO Group to be searched for Dymos Phases.

    Returns
    -------
    dict
        A dictionary mapping the absolute path of each Phase object in the given group to each
        Phase object.
    """
    phase_paths = {}
    if isinstance(sys, Phase):
        phase_paths[sys.pathname] = sys
    elif isinstance(sys, om.Group):
        for subsys in sys._loc_subsys_map:
            phase_paths.update(find_phases(getattr(sys, subsys)))
    return phase_paths


def re_interpolate_solution(problem, previous_solution):
    """
    Populate a guess for the given problem involving Dymos Phases by interpolating results
    from the previous solution.

    Parameters
    ----------
    problem : om.Problem
        An OpenMDAO Problem object which contains one or more Dymos Phases.
    previous_solution : dict
        A dictionary with key 'inputs' mapped to the output of problem.model.list_inputs for
        a previous iteration, and key 'outputs' mapped to the output of prob.model.list_outputs.
        Both list_inputs and list_outputs should be called with `units=True` and `prom_names=True`.
    """
    phase_paths = find_phases(problem.model)

    if not phase_paths:
        return

    prev_inputs = {v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in previous_solution['inputs']}
    prev_outputs = {v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in previous_solution['outputs']}

    phase_io = {'inputs': problem.model.list_inputs(out_stream=None, units=True, prom_name=True),
                'outputs': problem.model.list_outputs(out_stream=None, units=True, prom_name=True)}

    phase_inputs = {v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in phase_io['inputs']}
    phase_outputs = {v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in phase_io['outputs']}

    for phase_abs_path, phase in phase_paths.items():
        phase_name = phase_abs_path.split('.')[-1]

        # Get the initial time and duration from the previous result and set them into the new phase.
        ti_path = [s for s in phase_outputs if s.endswith(f'{phase_name}.t_initial')][0]
        t_initial = prev_outputs[ti_path]['value']
        t_initial_units = prev_outputs[ti_path]['units']

        td_path = [s for s in phase_outputs if s.endswith(f'{phase_name}.t_duration')][0]
        t_duration = prev_outputs[td_path]['value']
        t_duration_units = prev_outputs[td_path]['units']

        prev_time_path = [s for s in prev_outputs if s.endswith(f'{phase_name}.timeseries.time')][0]
        prev_time = prev_outputs[prev_time_path]['value']

        problem.set_val(ti_path, t_initial, units=t_initial_units)
        problem.set_val(td_path, t_duration, units=t_duration_units)

        # TODO: set the previous values of the phase and trajectory design parameters and polynomial controls

        # Interpolate the timeseries state outputs from the previous solution onto the new grid.
        for state_name, options in phase.state_options.items():
            state_path = [s for s in phase_outputs if s.endswith(f'{phase_name}.states:{state_name}')][0]
            prev_state_path = [s for s in prev_outputs if s.endswith(f'{phase_name}.timeseries.states:{state_name}')][0]
            prev_state_val = prev_outputs[prev_state_path]['value']
            prev_state_units = prev_outputs[prev_state_path]['units']
            problem.set_val(state_path,
                            phase.interpolate(xs=prev_time, ys=prev_state_val,
                                              nodes='state_input', kind='slinear'),
                            units=prev_state_units)

        # Interpolate the timeseries control outputs from the previous solution onto the new grid.
        for control_name, options in phase.control_options.items():
            control_path = [s for s in phase_outputs if s.endswith(f'{phase_name}.controls:{control_name}')][0]
            prev_control_path = [s for s in prev_outputs
                                 if s.endswith(f'{phase_name}.timeseries.controls:{control_name}')][0]
            prev_control_val = prev_outputs[prev_control_path]['value']
            prev_control_units = prev_outputs[prev_control_path]['units']
            problem.set_val(control_path,
                            phase.interpolate(xs=prev_time, ys=prev_control_val,
                                              nodes='control_input', kind='slinear'),
                            units=prev_control_units)
