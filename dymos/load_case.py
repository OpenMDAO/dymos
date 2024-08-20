import numpy as np

import openmdao.api as om
from openmdao.recorders.case import Case
from .phase import AnalyticPhase, Phase
from .trajectory import Trajectory
from openmdao.utils.om_warnings import issue_warning, warn_deprecation


def find_phases(sys):
    """
    Finds all instances of Dymos Phases within the given system, and returns them as a dictionary.

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
        for sub in sys.system_iter(recurse=False):
            phase_paths.update(find_phases(sub))
    return phase_paths


def find_trajectories(sys):
    """
    Finds all instances of Dymos Trajectories within the given system, and returns them as a dictionary.
    They are keyed by promoted name if use_prom_path=True, otherwise they are keyed by their
    absolute name.

    Parameters
    ----------
    sys : om.Group
        The OpenMDAO Group to be searched for Dymos Phases.

    Returns
    -------
    dict
        A dictionary mapping the absolute path of each Trajectory object in the given group to each
        Phase object.
    """
    traj_paths = {}
    if isinstance(sys, Trajectory):
        traj_paths[sys.pathname] = sys
    elif isinstance(sys, om.Group):
        for sub in sys.system_iter(recurse=False):
            traj_paths.update(find_trajectories(sub))
    return traj_paths


def load_case(problem, previous_solution, deprecation_warning=True):
    """
    Populate a guess for the given problem involving Dymos Phases by interpolating results
    from the previous solution.

    Parameters
    ----------
    problem : om.Problem
        An OpenMDAO Problem object which contains one or more Dymos Phases.
    previous_solution : dict [or Case]
        A dictionary with key 'inputs' mapped to the output of problem.model.list_inputs for
        a previous iteration, and key 'outputs' mapped to the output of prob.model.list_outputs.
        Both list_inputs and list_outputs should be called with `units=True` and `prom_names=True`.
    deprecation_warning : bool
        When False, no deprecation warning will be issued, otherwise warning will be issued.
        (defaults to True)
    """
    if deprecation_warning:
        warn_deprecation("The Dymos load_case method is deprecated for OpenMDAO 3.28.0 and later, "
                         "the load_case method on Problem should be used instead.")

    # allow old style arguments using a Case or OpenMDAO problem instead of dictionary
    assert (isinstance(previous_solution, Case) or isinstance(previous_solution, dict))
    if isinstance(previous_solution, Case):
        case = previous_solution
        previous_solution = {'inputs': case.list_inputs(out_stream=None, units=True, prom_name=True),
                             'outputs': case.list_outputs(out_stream=None, units=True, prom_name=True)}

    traj_paths = find_trajectories(problem.model)
    phase_paths = find_phases(problem.model)

    if not phase_paths and not traj_paths:
        return

    prev_vars = {}
    prev_vars.update({v['prom_name']: {'val': v['val'], 'units': v['units'], 'abs_name': k}
                      for k, v in previous_solution['inputs']})
    prev_vars.update({v['prom_name']: {'val': v['val'], 'units': v['units'], 'abs_name': k}
                      for k, v in previous_solution['outputs']})

    problem.final_setup()  # make sure list_inputs and list_outputs can work

    phase_io = {'inputs': problem.model.list_inputs(units=True, prom_name=True, out_stream=None),
                'outputs': problem.model.list_outputs(units=True, prom_name=True, out_stream=None)}

    phase_vars = {}
    phase_vars.update({v['prom_name']: {'val': v['val'], 'units': v['units'], 'abs_name': k}
                       for k, v in phase_io['inputs']})
    phase_vars.update({v['prom_name']: {'val': v['val'], 'units': v['units'], 'abs_name': k}
                       for k, v in phase_io['outputs']})

    for traj_abs_path, traj in traj_paths.items():
        traj_name = traj_abs_path.rpartition('.')[-1]
        for param_name in traj.parameter_options:
            prev_match = [s for s in prev_vars if s.endswith(f'{traj_name}.parameters:{param_name}')]
            if prev_match:
                # In previous outputs
                prev_data = prev_vars[prev_match[0]]
                prev_val = prev_data['val']
                prev_units = prev_data['units']
            else:
                raise Warning(f'Unable to find a value for {traj_name}.parameters:{param_name} in the restart file.')
            problem.set_val(f'{traj.pathname}.parameters:{param_name}', prev_val[0, ...], units=prev_units)

    for phase_abs_path, phase in phase_paths.items():
        phase_name = phase_abs_path.rpartition('.')[-1]

        # Get the initial time and duration from the previous result and set them into the new phase.
        try:
            integration_name = phase.time_options['name']
            prev_time_path = [s for s in prev_vars if s.endswith(f'{phase_name}.timeseries.{integration_name}')][0]
        except IndexError:
            continue

        prev_time_val = prev_vars[prev_time_path]['val']
        t_initial = prev_time_val[0]
        t_duration = prev_time_val[-1] - prev_time_val[0]
        prev_time_val, unique_idxs = np.unique(prev_time_val, return_index=True)
        prev_time_units = prev_vars[prev_time_path]['units']

        if t_duration < 0:
            # Unique sorts the data. In reverse-time phases, we need to undo it.
            prev_time_val = np.flip(prev_time_val, axis=0)
            unique_idxs = np.flip(unique_idxs, axis=0)

        ti_path = [s for s in phase_vars.keys() if s.endswith(f'{phase_name}.t_initial')]
        if ti_path:
            problem.set_val(ti_path[0], t_initial, units=prev_time_units)

        td_path = [s for s in phase_vars.keys() if s.endswith(f'{phase_name}.t_duration')]
        if td_path:
            problem.set_val(td_path[0], t_duration, units=prev_time_units)

        # Interpolate the timeseries state outputs from the previous solution onto the new grid.
        if not isinstance(phase, AnalyticPhase):
            for state_name, options in phase.state_options.items():
                state_path = [s for s in phase_vars if s.endswith(f'{phase_name}.states:{state_name}')][0]
                prev_state_path = [s for s in prev_vars if s.endswith(f'{phase_name}.timeseries.{state_name}')][0]
                prev_state_val = prev_vars[prev_state_path]['val']
                prev_state_units = prev_vars[prev_state_path]['units']
                problem.set_val(state_path,
                                phase.interp(name=state_name,
                                             xs=prev_time_val,
                                             ys=prev_state_val[unique_idxs],
                                             kind='slinear'),
                                units=prev_state_units)

                init_val_path = [s for s in phase_vars if s.endswith(f'{phase_name}.initial_states:{state_name}')]
                if init_val_path:
                    problem.set_val(init_val_path[0], prev_state_val[0, ...], units=prev_state_units)

                if options['fix_final']:
                    warning_message = f"{phase_name}.states:{state_name} specifies 'fix_final=True'. " \
                                      f"If the given restart file has a" \
                                      f" different final value this will overwrite the user-specified value"
                    issue_warning(warning_message)

            # Interpolate the timeseries control outputs from the previous solution onto the new grid.
            for control_name, options in phase.control_options.items():
                control_path = [s for s in phase_vars if s.endswith(f'{phase_name}.controls:{control_name}')][0]
                prev_control_path = [s for s in prev_vars
                                     if s.endswith(f'{phase_name}.timeseries.{control_name}')][0]
                prev_control_val = prev_vars[prev_control_path]['val']
                prev_control_units = prev_vars[prev_control_path]['units']
                problem.set_val(control_path,
                                phase.interp(name=control_name,
                                             xs=prev_time_val,
                                             ys=prev_control_val[unique_idxs],
                                             kind='slinear'),
                                units=prev_control_units)
                if options['fix_final']:
                    warning_message = f"{phase_name}.controls:{control_name} specifies 'fix_final=True'. " \
                                      f"If the given restart file has a" \
                                      f" different final value this will overwrite the user-specified value"
                    issue_warning(warning_message)

        # Set the timeseries parameter outputs from the previous solution as the parameter value
        for param_name in phase.parameter_options:
            prev_match = [s for s in prev_vars if s.endswith(f'{phase_name}.parameters:{param_name}')]
            if prev_match:
                # In previous outputs
                prev_data = prev_vars[prev_match[0]]
                prev_param_val = prev_data['val']
                prev_param_units = prev_data['units']
                param_path = [s for s in phase_vars if s.endswith(f'{phase_name}.parameters:{param_name}')][0]
            else:
                continue
            problem.set_val(param_path, prev_param_val[0, ...], units=prev_param_units)
