import numpy as np
import openmdao.api as om
from openmdao.recorders.case import Case
from .phase.phase import Phase
from .utils.lgl import lgl


def _split_var_path(path):
    """
    Split the given OpenMDAO variable path into a system path and a variable name.

    Parameters
    ----------
    path : str
        The variable path to be split

    Returns
    -------
    sys_path : str
        The path to the system containing the given variable.
    var_name : str
        The name of the variable without the system path prepended to it.
    """
    *sys_path, var_name = path.split('.')
    sys_path = '.'.join(sys_path)
    return sys_path, var_name


def _get_parent_phase(problem, path):
    """
    Given a path in a problem instance, if the path represents a variable or system
    in a Dymos Phase, return that Phase object, otherwise return None.

    Parameters
    ----------
    problem : OpenMDAO Problem instance
        The problem instance to be searched for a Phase object.
    path : str
        The path whose parent Phase is sought.

    Returns
    -------
    Phase or None
        The Phase object containing the given Path, or None if the Phase does not belong to a Phase.

    """
    sys_path, var_name = _split_var_path(path)
    comps = sys_path.split('.')
    path = comps[0]
    for i in range(1, len(comps)):
        if isinstance(problem.model._get_subsystem(path), Phase):
            return problem.model._get_subsystem(path)
    return None


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

    phase_paths = find_phases(problem.model)

    if not phase_paths:
        return

    prev_outputs = {v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in previous_solution['outputs']}

    problem.final_setup()  # make sure list_inputs and list_outputs can work

    phase_io = {'inputs': problem.model.list_inputs(out_stream=None, units=True, prom_name=True),
                'outputs': problem.model.list_outputs(out_stream=None, units=True, prom_name=True)}

    phase_inputs = {v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in phase_io['inputs']}
    phase_outputs = {v['prom_name']: {'value': v['value'], 'units': v['units']} for k, v in phase_io['outputs']}

    for phase_abs_path, phase in phase_paths.items():
        phase_name = phase_abs_path.split('.')[-1]

        # Get the initial time and duration from the previous result and set them into the new phase.
        t_path = [s for s in phase_outputs if s.endswith(f'{phase_name}.timeseries.time_phase')][0]

        t_initial = prev_outputs[t_path]['value'][0]
        t_initial_units = prev_outputs[t_path]['units']

        t_duration = prev_outputs[t_path]['value'][-1] - prev_outputs[t_path]['value'][0]
        t_duration_units = t_initial_units

        prev_time_path = [s for s in prev_outputs if s.endswith(f'{phase_name}.timeseries.time')][0]
        prev_time = prev_outputs[prev_time_path]['value']

        # initial time and duration may not be present if a simulation was loaded
        ti_path = [s for s in phase_outputs if s.endswith(f'{phase_name}.t_initial')][0]
        td_path = [s for s in phase_outputs if s.endswith(f'{phase_name}.t_duration')][0]
        problem.set_val(ti_path, t_initial, units=t_initial_units)
        problem.set_val(td_path, t_duration, units=t_duration_units)

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

        # Set the output polynomial control outputs from the previous solution as the value
        for polynomial_control_name, options in phase.polynomial_control_options.items():
            polynomial_control_path = [s for s in phase_outputs if
                                       s.endswith(f'{phase_name}.polynomial_controls:{polynomial_control_name}')][0]
            prev_polynomial_control_path = [s for s in prev_outputs if
                                            s.endswith(f'{phase_name}.'
                                                       f'polynomial_controls:{polynomial_control_name}')][0]
            prev_polynomial_control_val = prev_outputs[prev_polynomial_control_path]['value']
            prev_polynomial_control_units = prev_outputs[prev_polynomial_control_path]['units']
            problem.set_val(polynomial_control_path, prev_polynomial_control_val,
                            units=prev_polynomial_control_units)

        # Set the timeseries parameter outputs from the previous solution as the parameter value
        for parameter_name, options in phase.parameter_options.items():
            parameter_path = [s for s in phase_inputs if s.endswith(f'{phase_name}.parameters:{parameter_name}')][0]
            prev_parameter_path = [s for s in prev_outputs if
                                   s.endswith(f'{phase_name}.timeseries.parameters:{parameter_name}')][0]
            prev_parameter_val = prev_outputs[prev_parameter_path]['value'][0, ...]
            prev_parameter_units = prev_outputs[prev_parameter_path]['units']

            problem.set_val(parameter_path, prev_parameter_val, units=prev_parameter_units)
