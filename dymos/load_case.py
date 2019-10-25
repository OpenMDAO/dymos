import numpy as np

from .phase.phase import Phase


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


def load_case(problem, case):
    """
    Pull all input and output variables from a case into the model, taking special care to
    interpolate any values that are within Dymos phases.

    Parameters
    ----------
    problem : OpenMDAO Problem instance
        The Problem to be populated with data from the given Case.
    case : Case object or OpenMDAO Problem instance
        A Case from a CaseRecorder file.
    """
    inputs = case.inputs if case.inputs is not None else None
    outputs = case.outputs if case.outputs is not None else None
    vars = dict(inputs)
    vars.update(outputs)

    phases_in_case = set()
    phase_inputs = {}
    phase_outputs = {}

    if inputs:
        for name in inputs.absolute_names():
            if name not in problem.model._var_abs_names['input']:
                continue

            parent_phase = _get_parent_phase(problem, name)
            if parent_phase:
                if parent_phase not in phase_inputs:
                    phase_inputs[parent_phase] = []
                phase_inputs[parent_phase].append(name)
                phases_in_case.add(parent_phase)
            else:
                problem[name] = inputs[name]

    if outputs:
        for name in outputs.absolute_names():
            if name not in problem.model._var_abs_names['output']:
                continue

            parent_phase = _get_parent_phase(problem, name)
            if parent_phase:
                if parent_phase not in phase_outputs:
                    phase_outputs[parent_phase] = []
                    phase_outputs[parent_phase].append(name)
                phases_in_case.add(parent_phase)
            else:
                problem[name] = outputs[name]

    for phase in phases_in_case:
        phase_path = phase.pathname
        t_all = outputs[f'{phase_path}.timeseries.time']

        problem[f'{phase_path}.t_initial'] = outputs[f'{phase_path}.t_initial']
        problem[f'{phase_path}.t_duration'] = outputs[f'{phase_path}.t_duration']

        for state_name, options in phase.state_options.items():
            state_all = outputs[f'{phase_path}.timeseries.states:{state_name}']
            problem[f'{phase_path}.states:{state_name}'] = \
                phase.interpolate(t_all, state_all, nodes='state_input')

        for control_name, options in phase.control_options.items():
            control_all = outputs[f'{phase_path}.timeseries.controls:{control_name}']
            problem[f'{phase_path}.controls:{control_name}'] = \
                phase.interpolate(t_all, control_all, nodes='control_input')

        for control_name, options in phase.polynomial_control_options.items():
            control_all = outputs[f'{phase_path}.timeseries.polynomial_controls:{control_name}']
            problem[f'{phase_path}.polynomial_controls:{control_name}'] = \
                phase.interpolate(t_all, control_all, nodes='control_input')

        for param_name, options in phase.design_parameter_options.items():
            param_val = outputs[f'{phase_path}.design_parameters:{param_name}']
            problem[f'{phase_path}.design_parameters:{param_name}'] = param_val

        for param_name, options in phase.input_parameter_options.items():
            param_val = outputs[f'{phase_path}.input_parameters:{param_name}']
            problem[f'{phase_path}.input_parameters:{param_name}'] = param_val

        for var_path in phase_inputs[phase] + phase_outputs[phase]:
            if f'{phase_path}.rhs_all' in var_path:
                val_all = vars[var_path]
                var_shape = problem[var_path].shape
                problem[var_path] = np.reshape(phase.interpolate(t_all, val_all, nodes='all'), var_shape)

            if f'{phase_path}.rhs_disc' in var_path:
                t_disc = t_all[phase.options['transcription'].grid_data.subset_node_indices['state_disc']]
                val_disc = vars[var_path]
                var_shape = problem[var_path].shape
                problem[var_path] = np.reshape(phase.interpolate(t_disc, val_disc, nodes='state_disc'), var_shape)

            if f'{phase_path}.rhs_col' in var_path:
                t_col = t_all[phase.options['transcription'].grid_data.subset_node_indices['col']]
                val_col = vars[var_path]
                var_shape = problem[var_path].shape
                problem[var_path] = np.reshape(phase.interpolate(t_col, val_col, nodes='col'), var_shape)
