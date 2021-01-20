import openmdao.api as om
from openmdao.recorders.case import Case
from .phase.phase import Phase
from .trajectory import Trajectory


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


def load_case(problem, previous_solution):
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
    """

    # allow old style arguments using a Case or OpenMDAO problem instead of dictionary
    assert(isinstance(previous_solution, Case) or isinstance(previous_solution, dict))
    if isinstance(previous_solution, Case):
        case = previous_solution
        previous_solution = {'inputs': case.list_inputs(out_stream=None, units=True, prom_name=True),
                             'outputs': case.list_outputs(out_stream=None, units=True, prom_name=True)}

    traj_paths = find_trajectories(problem.model)
    phase_paths = find_phases(problem.model)

    if not phase_paths and not traj_paths:
        return

    prev_vars = {}
    prev_vars.update({v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in previous_solution['inputs']})
    prev_vars.update({v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in previous_solution['outputs']})

    problem.final_setup()  # make sure list_inputs and list_outputs can work

    phase_io = {'inputs': problem.model.list_inputs(out_stream=None, units=True, prom_name=True),
                'outputs': problem.model.list_outputs(out_stream=None, units=True, prom_name=True)}

    phase_vars = {}
    phase_vars.update({v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in phase_io['inputs']})
    phase_vars.update({v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in phase_io['outputs']})

    for traj_abs_path, traj in traj_paths.items():
        traj_name = traj_abs_path.split('.')[-1]
        for param_name, options in traj.parameter_options.items():
            prev_match = [s for s in prev_vars if s.endswith(f'{traj_name}.parameters:{param_name}')]
            if prev_match:
                # In previous outputs
                prev_data = prev_vars[prev_match[0]]
                prev_val = prev_data['value']
                prev_units = prev_data['units']
            else:
                raise Warning(f'Unable to find a value for {traj_name}.parameters:{param_name} in the restart file.')
            problem.set_val(f'{traj.pathname}.parameters:{param_name}', prev_val[0, ...], units=prev_units)

    for phase_abs_path, phase in phase_paths.items():
        phase_name = phase_abs_path.split('.')[-1]

        # Get the initial time and duration from the previous result and set them into the new phase.
        prev_time_path = [s for s in prev_vars if s.endswith(f'{phase_name}.timeseries.time')][0]

        prev_time_val = prev_vars[prev_time_path]['value']
        prev_time_units = prev_vars[prev_time_path]['units']

        t_initial = prev_time_val[0]
        t_duration = prev_time_val[-1] - prev_time_val[0]

        ti_path = [s for s in phase_vars.keys() if s.endswith(f'{phase_name}.t_initial')]
        if ti_path:
            problem.set_val(ti_path[0], t_initial, units=prev_time_units)

        td_path = [s for s in phase_vars.keys() if s.endswith(f'{phase_name}.t_duration')]
        if td_path:
            problem.set_val(td_path[0], t_duration, units=prev_time_units)

        # Interpolate the timeseries state outputs from the previous solution onto the new grid.
        for state_name, options in phase.state_options.items():
            state_path = [s for s in phase_vars if s.endswith(f'{phase_name}.states:{state_name}')][0]
            prev_state_path = [s for s in prev_vars if s.endswith(f'{phase_name}.timeseries.states:{state_name}')][0]
            prev_state_val = prev_vars[prev_state_path]['value']
            prev_state_units = prev_vars[prev_state_path]['units']
            problem.set_val(state_path,
                            phase.interpolate(xs=prev_time_val, ys=prev_state_val,
                                              nodes='state_input', kind='slinear'),
                            units=prev_state_units)

        # Interpolate the timeseries control outputs from the previous solution onto the new grid.
        for control_name, options in phase.control_options.items():
            control_path = [s for s in phase_vars if s.endswith(f'{phase_name}.controls:{control_name}')][0]
            prev_control_path = [s for s in prev_vars
                                 if s.endswith(f'{phase_name}.timeseries.controls:{control_name}')][0]
            prev_control_val = prev_vars[prev_control_path]['value']
            prev_control_units = prev_vars[prev_control_path]['units']
            problem.set_val(control_path,
                            phase.interpolate(xs=prev_time_val, ys=prev_control_val,
                                              nodes='control_input', kind='slinear'),
                            units=prev_control_units)

        # Set the output polynomial control outputs from the previous solution as the value
        for pc_name, options in phase.polynomial_control_options.items():
            pc_path = [s for s in phase_vars if
                       s.endswith(f'{phase_name}.polynomial_controls:{pc_name}')][0]
            prev_pc_path = [s for s in prev_vars if s.endswith(f'{phase_name}.polynomial_controls:{pc_name}')][0]
            prev_pc_val = prev_vars[prev_pc_path]['value']
            prev_pc_units = prev_vars[prev_pc_path]['units']
            problem.set_val(pc_path, prev_pc_val, units=prev_pc_units)

        # Set the timeseries parameter outputs from the previous solution as the parameter value
        for param_name, options in phase.parameter_options.items():
            prev_match = [s for s in prev_vars if s.endswith(f'{phase_name}.parameters:{param_name}')]
            if prev_match:
                # In previous outputs
                prev_data = prev_vars[prev_match[0]]
                prev_param_val = prev_data['value']
                prev_param_units = prev_data['units']
                param_path = [s for s in phase_vars if s.endswith(f'{phase_name}.parameters:{param_name}')][0]
            else:
                continue
            problem.set_val(param_path, prev_param_val[0, ...], units=prev_param_units)
