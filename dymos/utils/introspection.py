from collections.abc import Iterable
import fnmatch
import re

import openmdao.api as om
import numpy as np
from openmdao.utils.array_utils import shape_to_len
from dymos.utils.misc import _unspecified
from ..phase.options import StateOptionsDictionary, TimeseriesOutputOptionsDictionary
from .misc import get_rate_units


def classify_var(var, time_options, state_options, parameter_options, control_options,
                 polynomial_control_options, timeseries_options=None):
    """
    Classifies a variable of the given name or path.

    This method searches for it as a time variable, state variable,
    control variable, or parameter.  If it is not found to be one
    of those variables, it is assumed to be the path to a variable
    relative to the top of the ODE system for the phase.

    Parameters
    ----------
    var : str
        The name of the variable to be classified.
    time_options : OptionsDictionary
        Time options for the phase.
    state_options : dict of {str: OptionsDictionary}
        For each state variable, a dictionary of its options, keyed by name.
    parameter_options : dict of {str: OptionsDictionary}
        For each parameter, a dictionary of its options, keyed by name.
    control_options : dict of {str: OptionsDictionary}
        For each control variable, a dictionary of its options, keyed by name.
    polynomial_control_options : dict of {str: OptionsDictionary}
        For each polynomial variable, a dictionary of its options, keyed by name.
    timeseries_options : {str: OptionsDictionary}
        For each timeseries, a dictionary of its options, keyed by name.

    Returns
    -------
    str
        The classification of the given variable, which is one of
        'time', 't_phase', 'state', 'input_control', 'indep_control', 'control_rate',
        'control_rate2', 'input_polynomial_control', 'indep_polynomial_control',
        'polynomial_control_rate', 'polynomial_control_rate2', 'parameter',
        or 'ode'.
    """
    if var == time_options['name']:
        return 't'
    elif var == 't_phase':
        return 't_phase'
    elif var in state_options:
        return 'state'
    elif var in control_options:
        if control_options[var]['opt']:
            return 'indep_control'
        else:
            return 'input_control'
    elif var in polynomial_control_options:
        if polynomial_control_options[var]['opt']:
            return 'indep_polynomial_control'
        else:
            return 'input_polynomial_control'
    elif var in parameter_options:
        return 'parameter'
    elif var.endswith('_rate'):
        if var[:-5] in control_options:
            return 'control_rate'
        elif var[:-5] in polynomial_control_options:
            return 'polynomial_control_rate'
    elif var.endswith('_rate2'):
        if var[:-6] in control_options:
            return 'control_rate2'
        elif var[:-6] in polynomial_control_options:
            return 'polynomial_control_rate2'
    elif timeseries_options is not None:
        for timeseries in timeseries_options:
            if var in timeseries_options[timeseries]['outputs']:
                if timeseries_options[timeseries]['outputs'][var]['is_expr']:
                    return 'timeseries_exec_comp_output'

    return 'ode'


def get_promoted_vars(ode, iotypes, metadata_keys=None, get_remote=True):
    """
    Returns a dictionary mapping the promoted names of all inputs in a system to their associated metadata.

    Parameters
    ----------
    ode : openmdao.core.System
        The system from which the promoted inputs or outputs are being retrieved.
    iotypes : str or tuple
        One of 'input' or 'output', or a tuple of both.
    metadata_keys : Iterable or None
        Additional metadata requested for the variables. By default returns metadata available on 'allprocs'.  See
        openmdao.core.System.get_io_metadata for more information.
    get_remote : bool
        If True, include IO not local to this proc.

    Returns
    -------
    dict
        A dictionary mapping the promoted names of inputs in the system to their associated metadata.
    """
    _iotypes = (iotypes,) if isinstance(iotypes, str) else iotypes
    return {opts['prom_name']: opts for opts in ode.get_io_metadata(iotypes=_iotypes, get_remote=get_remote,
                                                                    metadata_keys=metadata_keys).values()}


def get_targets(ode, name, user_targets, control_rates=False):
    """
    Return the targets of a variable in a given ODE system.

    If the targets of the variable is _unspecified, and the name is a top level input name
    in the ODE, then the values are automatically connected to that top-level input.
    If _unspecified and not a top-level input of the ODE, no connection is made.
    If targets is explicitly None, then no connection is made.
    Otherwise, if the user specified some other string or sequence of strings as targets, then
    those are returned.

    Parameters
    ----------
    ode : openmdao.core.System or dict
        The OpenMDAO system which serves as the ODE for dymos, or a dictionary of promoted ODE input names and
        their associated metadata. If a System, it should already have had its setup and configure methods called.
    name : str
        The name of the variable whose targets are desired.
    user_targets : str or None or Sequence or _unspecified
        Targets for the variable as given by the user.
    control_rates : bool or int
        If True, search for the target of the variable with the given name.  If 1, search for
        the first rate of the variable '{control_name}_rate', and if 2, search for the second
        derivative of the variable '{control_name}_rate2'.

    Returns
    -------
    list
        The target inputs of the variable in the ODE, as a list.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    if isinstance(ode, dict):
        ode_inputs = ode
    else:
        ode_inputs = get_promoted_vars(ode, iotypes=('input',))
    if user_targets is _unspecified:
        if name in ode_inputs and control_rates not in {1, 2}:
            return [name]
        elif control_rates == 1 and f'{name}_rate' in ode_inputs:
            return [f'{name}_rate']
        elif control_rates == 2 and f'{name}_rate2' in ode_inputs:
            return [f'{name}_rate2']
    elif user_targets:
        if isinstance(user_targets, str):
            return [user_targets]
        else:
            return user_targets

    return []


def _configure_constraint_introspection(phase):
    """
    Modify constraint options in-place using introspection of the phase and its ODE.

    Parameters
    ----------
    phase : Phase
        The phase object whose boundary and path constraints are to be introspected.
    """
    for constraint_type, constraints in [('initial', phase._initial_boundary_constraints),
                                         ('final', phase._final_boundary_constraints),
                                         ('path', phase._path_constraints)]:
        for con in constraints:
            time_units = phase.time_options['units']

            # Determine the path to the variable which we will be constraining
            var = con['name']
            var_type = phase.classify_var(var)

            if var != con['constraint_name'] is not None and var_type != 'ode':
                om.issue_warning(f"Option 'constraint_name' on {constraint_type} constraint {var} is only "
                                 f"valid for ODE outputs. The option is being ignored.", om.UnusedOptionWarning)

            if var_type in {'time', 'time_phase'}:
                con['shape'] = (1,)
                con['units'] = time_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.{var_type}'

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                state_units = phase.state_options[var]['units']
                con['shape'] = state_shape
                con['units'] = state_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.states:{var}'

            elif var_type == 'parameter':
                param_shape = phase.parameter_options[var]['shape']
                param_units = phase.parameter_options[var]['units']
                con['shape'] = param_shape
                con['units'] = param_units if con['units'] is None else con['units']
                con['constraint_path'] = f'parameter_vals:{var}'

            elif var_type == 'indep_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                con['shape'] = control_shape
                con['units'] = control_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.controls:{var}'

            elif var_type == 'input_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                con['shape'] = control_shape
                con['units'] = control_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.controls:{var}'

            elif var_type == 'indep_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                con['shape'] = control_shape
                con['units'] = control_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.polynomial_controls:{var}'

            elif var_type == 'input_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                con['shape'] = control_shape
                con['units'] = control_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.polynomial_controls:{var}'

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                con['shape'] = control_shape
                con['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.control_rates:{var}'

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                con['shape'] = control_shape
                con['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.control_rates:{var}'

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                con['shape'] = control_shape
                con['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.polynomial_control_rates:{var}'

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                con['shape'] = control_shape
                con['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.polynomial_control_rates:{var}'

            else:
                # Failed to find variable, assume it is in the ODE. This requires introspection.
                ode = phase.options['transcription']._get_ode(phase)

                meta = get_source_metadata(ode, src=var, user_units=con['units'], user_shape=con['shape'])

                con['shape'] = meta['shape']
                con['units'] = meta['units']
                con['constraint_path'] = f'timeseries.{con["constraint_name"]}'


def configure_controls_introspection(control_options, ode, time_units='s'):
    """
    Modify control options in-place using introspection of the user-provided ODE.

    Parameters
    ----------
    control_options : dict of {str: ControlOptionsDictionary
        A dictionary keyed by control name containing the options for all controls to be applied
        to the ODE.
    ode : om.System
        An instantiated System that serves as the ODE to which the controls should be applied.
    time_units : str
        The units of time for the Phase.

    Raises
    ------
    ValueError
        If the control or one of its rates are connected to a variable that is tagged as static
        within the ODE.
    """
    ode_inputs = get_promoted_vars(ode, iotypes='input')
    for name, options in control_options.items():

        targets, shape, units, static_target = _get_targets_metadata(ode_inputs,
                                                                     name=name,
                                                                     user_targets=options['targets'],
                                                                     user_units=options['units'],
                                                                     user_shape=options['shape'],
                                                                     control_rate=False)
        options['targets'] = targets
        if targets:
            options['units'] = units
            options['shape'] = shape

        if static_target:
            raise ValueError(f"Control '{name}' cannot be connected to its targets because one "
                             f"or more targets are tagged with 'dymos.static_target'.")

        # Now check rate targets
        rate_targets, shape, units, static_target = _get_targets_metadata(ode_inputs,
                                                                          name=name,
                                                                          user_targets=options['rate_targets'],
                                                                          user_units=options['units'],
                                                                          user_shape=options['shape'],
                                                                          control_rate=1)
        options['rate_targets'] = rate_targets
        if options['units'] is _unspecified:
            options['units'] = time_units if units is None else f'{units}*{time_units}'
        options['shape'] = shape
        if static_target:
            raise ValueError(f"Control rate of '{name}' cannot be connected to its targets "
                             f"because one or more targets are tagged with 'dymos.static_target'.")

        # Now check rate2 targets
        rate2_targets, shape, units, static_target = _get_targets_metadata(ode_inputs,
                                                                           name=name,
                                                                           user_targets=options['rate2_targets'],
                                                                           user_units=options['units'],
                                                                           user_shape=options['shape'],
                                                                           control_rate=2)
        options['rate2_targets'] = rate2_targets
        if options['units'] is _unspecified:
            options['units'] = time_units if units is None else f'{units}*{time_units}**2'
        options['shape'] = shape
        if static_target:
            raise ValueError(f"Control rate2 of '{name}' cannot be connected to its targets "
                             f"because one or more targets are tagged with 'dymos.static_target'.")


def configure_parameters_introspection(parameter_options, ode):
    """
    Modify parameter options in-place using introspection of the user-provided ODE.

    Parameters
    ----------
    parameter_options : dict of {str: ParameterOptionsDictionary
        A dictionary keyed by parameter name containing the options for all parameters to be applied
        to the ODE. Options for 'targets', 'units', 'shape', and 'static_target' are modified in-place.
    ode : om.System
        An instantiated System that serves as the ODE to which the parameters should be applied.
    """
    ode_inputs = get_promoted_vars(ode, iotypes='input')

    for name, options in parameter_options.items():
        try:
            targets, shape, units, static_target = _get_targets_metadata(ode_inputs, name=name,
                                                                         user_targets=options['targets'],
                                                                         user_units=options['units'],
                                                                         user_shape=options['shape'],
                                                                         user_static_target=options['static_target'])
        except ValueError as e:
            raise ValueError(f'Parameter `{name}` has invalid target(s).\n{str(e)}') from e
        options['targets'] = targets
        options['units'] = units
        options['shape'] = shape
        options['static_target'] = static_target


def configure_time_introspection(time_options, ode):
    """
    Modify time options in-place using introspection of the user-provided ODE.

    Parameters
    ----------
    time_options : dict of {str: TimeOptionsDictionary
        A dictionary keyed by control name containing the options for all controls to be applied
        to the ODE. Options for 'targets', 'time_phase_targets', and 'units' are modified in-place.
    ode : om.System
        An instantiated System that serves as the ODE to which the controls should be applied.

    Raises
    ------
    ValueError
        If time or time_phase are connected to a variable that is tagged as static
        within the ODE.
    """
    ode_inputs = get_promoted_vars(ode, 'input')
    # time
    targets, shape, units, static_target = _get_targets_metadata(ode_inputs,
                                                                 name='time',
                                                                 user_targets=time_options['targets'],
                                                                 user_units=time_options['units'],
                                                                 user_shape=(1,))

    time_options['targets'] = targets
    time_options['units'] = units

    if static_target:
        raise ValueError(f"'time' cannot be connected to its targets because one "
                         f"or more targets are tagged with 'dymos.static_target'.")

    # time_phase
    targets, shape, units, static_target = _get_targets_metadata(ode_inputs,
                                                                 name='time_phase',
                                                                 user_targets=time_options['time_phase_targets'],
                                                                 user_units=time_options['units'],
                                                                 user_shape=(1,))

    time_options['time_phase_targets'] = targets

    if static_target:
        raise ValueError(f"'time_phase' cannot be connected to its targets because one "
                         f"or more targets are tagged with 'dymos.static_target'.")


def configure_states_introspection(state_options, time_options, control_options, parameter_options,
                                   polynomial_control_options, ode):
    """
    Modifies state options in-place, automatically determining 'targets', 'units', and 'shape' if necessary.

    The precedence rules for the state shape and units are as follows:
    1. If the user has specified units and shape in the state options, use those.
    2a. If the user has not specified shape, and targets exist, then pull the shape from the targets.
    2b. If the user has not specified shape and no targets exist, then pull the shape from the rate source.
    2c. If shape cannot be inferred, assume (1,)
    3a. If the user has not specified units, first try to pull units from a target
    3b. If there are no targets, pull units from the rate source and multiply by time units.

    Parameters
    ----------
    state_options : dict of {str: StateOptionsDictionary}
        The state variables to be configured.
    time_options : TimeOptionsDictionary
        The time options.
    control_options : dict of {str: ControlOptionsDictionary}
        The options for each control.
    parameter_options : dict of {str: ParameterOptionsDictionary}
        The options for each parameter.
    polynomial_control_options : dict of {str: PolynomialControlOptionsDictionary}
        The options for each polynomial control.
    ode : System
        The OpenMDAO system which provides the state rates as outputs.
    """
    time_units = time_options['units']
    ode_inputs = get_promoted_vars(ode, 'input')
    ode_outputs = get_promoted_vars(ode, 'output')

    for state_name, options in state_options.items():
        # Automatically determine targets of state if left _unspecified
        targets, tgt_shape, tgt_units, static_target = _get_targets_metadata(ode_inputs,
                                                                             state_name,
                                                                             user_targets=options['targets'],
                                                                             user_units=options['units'],
                                                                             user_shape=options['shape'])

        options['targets'] = targets
        if targets:
            options['shape'] = tgt_shape
            options['units'] = tgt_units

        # 3. Attempt rate-source introspection
        rate_src = options['rate_source']
        rate_src_type = classify_var(rate_src, time_options, state_options, parameter_options, control_options,
                                     polynomial_control_options)

        if rate_src_type in {'time', 'time_phase'}:
            rate_src_units = time_options['units']
            rate_src_shape = (1,)
        elif rate_src_type == 'state':
            rate_src_units = state_options[rate_src]['units']
            rate_src_shape = state_options[rate_src]['shape']
        elif rate_src_type in ['input_control', 'indep_control']:
            rate_src_units = control_options[rate_src]['units']
            rate_src_shape = control_options[rate_src]['shape']
        elif rate_src_type in ['input_polynomial_control', 'indep_polynomial_control']:
            rate_src_units = polynomial_control_options[rate_src]['units']
            rate_src_shape = polynomial_control_options[rate_src]['shape']
        elif rate_src_type == 'parameter':
            rate_src_units = parameter_options[rate_src]['units']
            rate_src_shape = parameter_options[rate_src]['shape']
        elif rate_src_type == 'control_rate':
            control_name = rate_src[:-5]
            control = control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=1)
            rate_src_shape = control['shape']
        elif rate_src_type == 'control_rate2':
            control_name = rate_src[:-6]
            control = control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=2)
            rate_src_shape = control['shape']
        elif rate_src_type == 'polynomial_control_rate':
            control_name = rate_src[:-5]
            control = polynomial_control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=1)
            rate_src_shape = control['shape']
        elif rate_src_type == 'polynomial_control_rate2':
            control_name = rate_src[:-6]
            control = polynomial_control_options[control_name]
            rate_src_units = get_rate_units(control['units'], time_units, deriv=2)
            rate_src_shape = control['shape']
        elif rate_src_type == 'ode':
            meta = get_source_metadata(ode_outputs, src=rate_src, user_units=options['units'], user_shape=options['shape'])
            rate_src_shape = meta['shape']
            rate_src_units = meta['units']
        else:
            rate_src_shape = (1,)
            rate_src_units = None

        if options['shape'] in {None, _unspecified}:
            options['shape'] = rate_src_shape

        if options['units'] is _unspecified:
            if rate_src_units is None:
                options['units'] = time_units
            else:
                if time_units is None:
                    raise RuntimeError(f'Unable to infer the units of state variable `{state_name}` from\n'
                                       f'the rate units because the time units of the phase are set to None.\n'
                                       f'Change the time units to something other than None, or explicitly\n'
                                       f'set the state units using one of the following options:\n'
                                       f'- Tag the state rate source `{rate_src}` with `dymos.state_units:{{units}}`\n'
                                       f'- Use the `set_state_options(\'{state_name}\', units={{units}})` method on the phase.')
                else:
                    options['units'] = f'{rate_src_units}*{time_units}'


def configure_analytic_states_introspection(state_options, ode):
    """
    Modifies state options in-place, automatically determining 'targets', 'units', and 'shape' if necessary.

    The precedence rules for the state shape and units are as follows:
    1. If the user has specified units and shape in the state options, use those.
    2a. If the user has not specified shape, and targets exist, then pull the shape from the targets.
    2b. If the user has not specified shape and no targets exist, then pull the shape from the rate source.
    2c. If shape cannot be inferred, assume (1,)
    3a. If the user has not specified units, first try to pull units from a target
    3b. If there are no targets, pull units from the rate source and multiply by time units.

    Parameters
    ----------
    state_options : dict of {str: StateOptionsDictionary}
        The state variables to be configured.
    ode : System
        The OpenMDAO system which provides the state rates as outputs.
    """
    ode_outputs = get_promoted_vars(ode, 'output')

    for state_name, options in state_options.items():
        # Automatically determine targets of state if left _unspecified
        source = options['source'] if options['source'] else state_name
        meta = get_source_metadata(ode_outputs, src=source, user_units=options['units'], user_shape=options['shape'])
        src_shape = meta['shape']
        src_units = meta['units']

        if 'dymos.static_output' in meta['tags']:
            raise RuntimeError(f'ODE output {source} is tagged with `dymos.static_output` and cannot be used as a '
                               f'state variable in an AnalyticPhase.')

        if options['shape'] in {None, _unspecified}:
            options['shape'] = src_shape

        if options['units'] is _unspecified:
            options['units'] = src_units


def configure_states_discovery(state_options, ode):
    """
    Searches phase output metadata for any declared states and adds them.

    Parameters
    ----------
    state_options : dict
        The dictionary of options for each state in the phase.
    ode : System
        The System instance providing the ODE for the phase.
    """
    out_meta = ode.get_io_metadata(iotypes='output', metadata_keys=['tags'],
                                   get_remote=True)

    for name, meta in out_meta.items():
        tags = meta['tags']
        prom_name = meta['prom_name']
        state = None
        for tag in sorted(tags):

            # Declared as rate_source.
            if tag.startswith('dymos.state_rate_source:') or tag.startswith('state_rate_source:'):
                state = tag.rpartition(':')[-1]
                if tag.startswith('state_rate_source:'):
                    msg = f"The tag '{tag}' has a deprecated format and will no longer work in " \
                          f"dymos version 2.0.0. Use 'dymos.state_rate_source:{state}' instead."
                    om.issue_warning(msg, category=om.OMDeprecationWarning)
                if state not in state_options:
                    state_options[state] = StateOptionsDictionary()
                    state_options[state]['name'] = state

                if state_options[state]['rate_source'] is not None:
                    if state_options[state]['rate_source'] != prom_name:
                        raise ValueError(f"rate_source has been declared twice for state "
                                         f"'{state}' which is tagged on '{name}'.")

                state_options[state]['rate_source'] = prom_name

            # Declares units for state.
            if tag.startswith('dymos.state_units:') or tag.startswith('state_units:'):
                tagged_state_units = tag.rpartition(':')[-1]
                if tag.startswith('state_units:'):
                    msg = f"The tag '{tag}' has a deprecated format and will no longer work in " \
                          f"dymos version 2.0.0. Use 'dymos.{tag}' instead."
                    om.issue_warning(msg, category=om.OMDeprecationWarning)
                if state is None:
                    raise ValueError(f"'{tag}' tag declared on '{prom_name}' also requires "
                                     f"that the 'dymos.state_rate_source:{tagged_state_units}' "
                                     f"tag be declared.")
                state_options[state]['units'] = tagged_state_units

    # Check over all existing states and make sure we aren't missing any rate sources.
    for name, options in state_options.items():
        if options['rate_source'] is None:
            raise ValueError(f"State '{name}' is missing a rate_source.")


def configure_analytic_states_discovery(state_options, ode):
    """
    Searches phase output metadata for any declared states and adds them.

    Parameters
    ----------
    state_options : dict
        The dictionary of options for each state in the phase.
    ode : System
        The System instance providing the ODE for the phase.
    """
    out_meta = ode.get_io_metadata(iotypes='output', metadata_keys=['tags'],
                                   get_remote=True)

    for name, meta in out_meta.items():
        tags = meta['tags']
        prom_name = meta['prom_name']
        state = None
        for tag in sorted(tags):

            # Declared as rate_source.
            if tag.startswith('dymos.state_source:'):
                state = tag.rpartition(':')[-1]
                if state not in state_options:
                    state_options[state] = StateOptionsDictionary()
                    state_options[state]['name'] = state

                if state_options[state]['source'] is not None:
                    if state_options[state]['source'] != prom_name:
                        raise ValueError(f"source has been declared twice for state "
                                         f"'{state}' which is tagged on '{name}'.")

                state_options[state]['source'] = prom_name

            # Declares units for state.
            if tag.startswith('dymos.state_units:') or tag.startswith('state_units:'):
                tagged_state_units = tag.rpartition(':')[-1]
                if state is None:
                    raise ValueError(f"'{tag}' tag declared on '{prom_name}' also requires "
                                     f"that the 'dymos.state_source:{tagged_state_units}' "
                                     f"tag be declared.")
                state_options[state]['units'] = tagged_state_units

    # Check over all existing states and make sure we aren't missing any rate sources.
    for name, options in state_options.items():
        if options['source'] is None:
            raise ValueError(f"State '{name}' is missing a source.")


def configure_timeseries_output_glob_expansion(phase):
    """
    Modify timeseries outputs in-place using introspection to expand any glob patterns.

    Parameters
    ----------
    phase : Phase
        The phase object whose boundary and path constraints are to be introspected.
    """
    transcription = phase.options['transcription']
    ode = transcription._get_ode(phase)
    ode_outputs = get_promoted_vars(ode, 'output')

    new_outputs = {}

    # Step 1: Expand globs
    for ts_name, ts_meta in phase._timeseries.items():

        explicit_requests = set([output['name'] for output in ts_meta['outputs'].values() if '*' not in output['name']])

        for output_name, output_options in ts_meta['outputs'].items():

            if '*' in output_name:
                matching_outputs = filter_outputs(output_name, ode_outputs)
                wildcard_units = {} if output_options['wildcard_units'] is None else output_options['wildcard_units']

                for op, meta in matching_outputs.items():
                    if op not in explicit_requests and 'dymos.static_output' not in meta['tags']:
                        new_output = TimeseriesOutputOptionsDictionary()
                        new_output['name'] = op
                        new_output['output_name'] = opname = op.split('.')[-1]
                        new_output['units'] = _unspecified if op not in wildcard_units else wildcard_units[op]
                        new_output['shape'] = _unspecified
                        if opname not in explicit_requests and opname in new_outputs:
                            raise RuntimeError(f'{phase.pathname}: The glob pattern `{output_name}` matches multiple '
                                               f'outputs in the ODE.\n'
                                               f'Add these outputs explicitly with unique output names using the\n'
                                               f'output_name argument to avoid this error.\n'
                                               f'Colliding names: {op}  {new_outputs[opname]["name"]}')
                        else:
                            new_outputs[new_output['output_name']] = new_output
            else:
                new_outputs[output_name] = output_options

        phase._timeseries[ts_name]['outputs'] = new_outputs


def configure_timeseries_output_introspection(phase):
    """
    Modify timeseries outputs in-place using introspection to find output units and shape.

    Parameters
    ----------
    phase : Phase
        The phase object whose timeseries outputs are to be introspected.
    """
    configure_timeseries_output_glob_expansion(phase)

    transcription = phase.options['transcription']

    for ts_name, ts_opts in phase._timeseries.items():

        not_found = set()

        for output_name, output_options in ts_opts['outputs'].items():
            if output_options['is_expr']:
                output_meta = configure_timeseries_expr_introspection(phase, ts_name, output_options)
            else:
                try:
                    output_meta = transcription._get_timeseries_var_source(output_options['name'],
                                                                           output_options['output_name'],
                                                                           phase=phase)
                except ValueError as e:
                    not_found.add(output_name)

            output_options['src'] = output_meta['src']
            output_options['src_idxs'] = output_meta['src_idxs']

            if output_options['shape'] in (None, _unspecified):
                output_options['shape'] = output_meta['shape']

            if output_options['units'] in (None, _unspecified):
                output_options['units'] = output_meta['units']

        if not_found:
            sorted_list = ', '.join(sorted(not_found))
            om.issue_warning(f'{phase.pathname}: The following timeseries outputs were requested but not found in the '
                             f'ODE: {sorted_list}')

        for s in not_found:
            ts_opts['outputs'].pop(s)


def configure_timeseries_expr_introspection(phase, timeseries_name, expr_options):
    """
    Introspect the timeseries expressions, add expressions to the exec comp, and make connections to the variables.

    Parameters
    ----------
    phase : Phase
        The phase object whose timeseries outputs are to be introspected.
    timeseries_name : str
        The name of the timeseries that the expr is to be added to.
    expr_options : dict
        Options for the expression to be added to the timeseries exec comp.

    Returns
    -------
    dict
        A dictionary containing the metadata for the source. This consists of shape, units, and tags.
    """
    timeseries_ec = phase._get_subsystem(f'{timeseries_name}.timeseries_exec_comp')
    transcription = phase.options['transcription']
    num_output_nodes = transcription._get_num_timeseries_nodes()

    expr = expr_options['name']
    expr_lhs = expr.split('=')[0].strip()
    if '.' in expr_lhs or ':' in expr_lhs:
        raise ValueError(f'{expr_lhs} is an invalid name for the timeseries LHS. Must use a valid variable name')
    expr_reduced = expr

    units = expr_options['units'] if expr_options['units'] is not _unspecified else None
    shape = expr_options['shape'] if expr_options['shape'] is not _unspecified else (1,)

    abs_names = [x.strip() for x in re.findall(re.compile(r'([_a-zA-Z]\w*[ ]*\(?:?[.]?)'), expr)
                 if not x.endswith('(') and not x.endswith(':')]
    for name in abs_names:
        if name.endswith('.'):
            idx = abs_names.index(name)
            abs_names[idx + 1] = name + abs_names[idx + 1]
            abs_names.remove(name)
    for name in abs_names:
        var_name = name.split('.')[-1]
        if var_name not in phase.timeseries_ec_vars.keys():
            phase.timeseries_ec_vars[var_name] = {'added_to_ec': False, 'abs_name': name}
    expr_vars = {}
    expr_kwargs = expr_options['expr_kwargs']
    kwargs_rel_shape = {}
    meta = {'units': units, 'shape': shape}

    for var in phase.timeseries_ec_vars:
        if var in ('output_name', 'units', 'shape', 'timeseries'):
            raise RuntimeError(f'{phase.pathname}: Timeseries expression `{expr}` uses the following disallowed '
                               f'variable name in its definition: \'{var}\'.')

        if phase.timeseries_ec_vars[var]['added_to_ec']:
            continue
        elif var == expr_lhs:
            var_units = units
            var_shape = shape
            meta['src'] = f'{timeseries_name}.timeseries_exec_comp.{var}'
            phase.timeseries_ec_vars[var]['added_to_ec'] = True
        else:
            try:
                expr_vars[var] = transcription._get_timeseries_var_source(var, output_name=var, phase=phase)
            except ValueError as e:
                expr_vars[var] = transcription._get_timeseries_var_source(phase.timeseries_ec_vars[var]['abs_name'],
                                                                          output_name=var, phase=phase)

            var_units = expr_vars[var]['units']
            var_shape = expr_vars[var]['shape']

        if var not in expr_kwargs:
            expr_kwargs[var] = {}

        if 'units' not in expr_kwargs[var]:
            expr_kwargs[var]['units'] = var_units
        elif var == expr_lhs:
            meta['units'] = expr_kwargs[var]['units']

        if 'shape' in expr_kwargs[var] and var != expr_lhs:
            om.issue_warning(f'{phase.pathname}: User-provided shape for timeseries expression input `{var}` ignored.\n'
                             f'Using automatically determined shape {(num_output_nodes,) + var_shape}.',
                             category=om.UnusedOptionWarning)
        elif 'shape' in expr_kwargs[var] and var == expr_lhs:
            if expr_options['shape'] is not _unspecified:
                om.issue_warning(f'{phase.pathname}: User-provided shape for timeseries expression output `{var}`'
                                 f'using both the\n`shape` keyword argument and the `{var}` keyword argument '
                                 f'dictionary.\nThe latter will take precedence.',
                                 category=om.SetupWarning)
            var_shape = expr_kwargs[var]['shape']
            meta['shape'] = var_shape

        expr_kwargs[var]['shape'] = var_shape
        kwargs_rel_shape[var] = {'units': expr_kwargs[var]['units'],
                                 'shape': (num_output_nodes,) + expr_kwargs[var]['shape']}

    for input_var in expr_vars:
        abs_name = phase.timeseries_ec_vars[input_var]['abs_name']
        if '.' in abs_name:
            expr_reduced = expr_reduced.replace(abs_name, input_var)

    timeseries_ec.add_expr(expr_reduced, **kwargs_rel_shape)

    for var, options in expr_vars.items():
        if phase.timeseries_ec_vars[var]['added_to_ec']:
            continue
        else:
            phase.connect(src_name=options['src'], tgt_name=f'{timeseries_name}.timeseries_exec_comp.{var}',
                          src_indices=options['src_idxs'])

        phase.timeseries_ec_vars[var]['added_to_ec'] = True

    meta['src_idxs'] = None

    return meta


def filter_outputs(patterns, sys):
    """
    Find all outputs of the given system that match one or more of the strings given in patterns.

    Parameters
    ----------
    patterns : str or Sequence
        A string or sequence of strings to be matched in the outputs of the given system.  These
        may include glob patterns.
    sys : System or dict
        The OpenMDAO system whose outputs are to be filtered, or a dictionary of outputs as returned by
        get_promoted_vars.

    Returns
    -------
    dict of {str: dict}
        A dictionary where the matching output names are the keys and the associated dict provides
        the 'units' and 'shapes' metadata.
    """
    outputs = sys if isinstance(sys, dict) else get_promoted_vars(sys, iotypes='output', metadata_keys=['shape', 'units', 'tags'])
    _patterns = [patterns] if isinstance(patterns, str) else patterns

    output_names = list(outputs.keys())
    filtered = set()
    results = {}

    for pattern in _patterns:
        filtered.update(fnmatch.filter(output_names, pattern))

    for var in filtered:
        results[var] = {'units': outputs[var]['units'], 'shape': outputs[var]['shape'], 'tags': outputs[var]['tags']}

    return results


def get_target_metadata(ode, name, user_targets=_unspecified, user_units=_unspecified,
                        user_shape=_unspecified, control_rate=False, user_static_target=_unspecified):
    """
    Return the targets of a state variable in a given ODE system.

    If the targets of the state is _unspecified, and the state name is a top level input name
    in the ODE, then the state values are automatically connected to that top-level input.
    If _unspecified and not a top-level input of the ODE, no connection is made.
    If targets is explicitly None, then no connection is made.
    Otherwise, if the user specified some other string or sequence of strings as targets, then
    those are returned.

    Parameters
    ----------
    ode : om.System or dict
        The OpenMDAO system which serves as the ODE for dymos, or a dictionary of inputs as returned by
        utils.introspection.get_promoted_vars.  If a system, it should already have had its setup and configure
        methods called.
    name : str
        The name of the variable whose targets are desired.
    user_targets : str or None or Sequence or _unspecified
        Targets for the variable as given by the user.
    user_units : str or None or _unspecified
        Units for the variable as given by the user.
    user_shape : None or Sequence or _unspecified
        Shape for the variable as given by the user.
    control_rate : bool
        When True, check for the control rate if the name is not in the ODE.
    user_static_target : bool or None or _unspecified
        When False, assume the shape of the target in the ODE includes the number of nodes as the
        first dimension.  If True, the connecting parameter does not need to be "fanned out" to
        connect to each node.  If _unspecified, attempt to resolve by the presence of a tag
        `dymos.static_target` on the target variable, which is the same as `static_target=True`.

    Returns
    -------
    shape : tuple
        The shape of the variable.  If not specified, shape is taken from the ODE targets.
    units : str
        The units of the variable.  If not specified, units are taken from the ODE targets.
    static_target : bool
        True if the target is static, otherwise False.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    ode_inputs = ode if isinstance(ode, dict) else get_promoted_vars(ode, iotypes='input')

    if user_targets is _unspecified:
        if name in ode_inputs:
            targets = [name]
        elif control_rate and f'{name}_rate' in ode_inputs:
            targets = [f'{name}_rate']
        elif control_rate and f'{name}_rate2' in ode_inputs:
            targets = [f'{name}_rate2']
        else:
            targets = []
    elif user_targets:
        if isinstance(user_targets, str):
            targets = [user_targets]
        else:
            targets = user_targets
    else:
        targets = []

    if user_units is _unspecified:
        target_units_set = {ode_inputs[tgt]['units'] for tgt in targets}
        if len(target_units_set) == 1:
            units = next(iter(target_units_set))
        else:
            raise ValueError(f'Unable to automatically assign units to {name}. '
                             f'Targets have multiple units: {target_units_set}. '
                             f'Either promote targets and use set_input_defaults to assign common '
                             f'units, or explicitly provide them to {name}.')
    else:
        units = user_units

    # Resolve whether the targets is static or dynamic
    static_target_tags = [tgt for tgt in targets if 'dymos.static_target' in ode_inputs[tgt]['tags']]
    if static_target_tags:
        static_target = True
        if not user_static_target:
            raise ValueError(f"User has specified 'static_target = False' for parameter {name},"
                             f"but one or more targets is tagged with "
                             f"'dymos.static_target': {' '.join(static_target_tags)}")
    else:
        if user_static_target is _unspecified:
            static_target = False
        else:
            static_target = user_static_target

    if user_shape in {None, _unspecified}:
        # Resolve target shape
        target_shape_set = {ode_inputs[tgt]['shape'] for tgt in targets}
        if len(target_shape_set) == 1:
            shape = next(iter(target_shape_set))
            if not static_target:
                if len(shape) == 1:
                    shape = (1,)
                else:
                    shape = shape[1:]
        elif len(target_shape_set) == 0:
            raise ValueError(f'Unable to automatically assign a shape to {name}.\n'
                             'Targets for this variable either do not exist or have no shape set.\n'
                             'The shape for this variable must be set explicitly via the '
                             '`shape=<tuple>` argument.')
        else:
            raise ValueError(f'Unable to automatically assign a shape to {name} based on targets. '
                             f'Targets have multiple shapes assigned: {target_shape_set}. '
                             f'Change targets such that all have common shapes.')
    else:
        shape = user_shape

    return shape, units, static_target


def _get_targets_metadata(ode, name, user_targets=_unspecified, user_units=_unspecified,
                          user_shape=_unspecified, control_rate=False, user_static_target=_unspecified):
    """
    Return the targets of a variable in a given ODE system and their metadata.

    If the targets of the state is _unspecified, and the state name is a top level input name
    in the ODE, then the state values are automatically connected to that top-level input.
    If _unspecified and not a top-level input of the ODE, no connection is made.
    If targets is explicitly None, then no connection is made.
    Otherwise, if the user specified some other string or sequence of strings as targets, then
    those are returned.

    Parameters
    ----------
    ode : om.System or dict
        The OpenMDAO system which serves as the ODE for dymos, or a dictionary of the ODE inputs and their metadata
        from a previous call to _get_targets_metadata. If given as the ODE system, it should already have had its setup
        and configure methods called.
    name : str
        The name of the variable whose targets are desired.
    user_targets : str or None or Sequence or _unspecified
        Targets for the variable as given by the user.
    user_units : str or None or _unspecified
        Units for the variable as given by the user.
    user_shape : None or Sequence or _unspecified
        Shape for the variable as given by the user.
    control_rate : bool
        When True, check for the control rate if the name is not in the ODE.
    user_static_target : bool or None or _unspecified
        When False, assume the shape of the target in the ODE includes the number of nodes as the
        first dimension.  If True, the connecting parameter does not need to be "fanned out" to
        connect to each node.  If _unspecified, attempt to resolve by the presence of a tag
        `dymos.static_target` on the target variable, which is the same as `static_target=True`.

    Returns
    -------
    targets : list
        The target inputs of the variable in the ODE, as a list.
    shape : tuple
        The shape of the variable.  If not specified, shape is taken from the ODE targets.
    units : str
        The units of the variable.  If not specified, units are taken from the ODE targets.
    static_target : bool
        True if the target is static, otherwise False.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    ode_inputs = ode if isinstance(ode, dict) else get_promoted_vars(ode, iotypes='input')

    targets = get_targets(ode_inputs, name, user_targets=user_targets, control_rates=control_rate)

    if not targets:
        return targets, user_shape, user_units, False

    for tgt in targets:
        if tgt not in ode_inputs:
            raise ValueError(f"No such ODE input: '{tgt}'.")

    if user_units is _unspecified:
        target_units_set = {ode_inputs[tgt]['units'] for tgt in targets}
        if len(target_units_set) == 1:
            units = next(iter(target_units_set))
        else:
            raise ValueError(f'Unable to automatically assign units to {name}. '
                             f'Targets have multiple units: {target_units_set}. '
                             f'Either promote targets and use set_input_defaults to assign common '
                             f'units, or explicitly provide them to {name}.')
    else:
        units = user_units

    # Resolve whether the targets is static or dynamic
    static_target_tags = [tgt for tgt in targets if 'dymos.static_target' in ode_inputs[tgt]['tags']]
    if static_target_tags:
        static_target = True
        if not user_static_target:
            raise ValueError(f"User has specified 'static_target = False' for parameter {name},"
                             f"but one or more targets is tagged with "
                             f"'dymos.static_target': {' '.join(static_target_tags)}")
    else:
        if user_static_target is _unspecified:
            static_target = False
        else:
            static_target = user_static_target

    if user_shape in {None, _unspecified}:
        # Resolve target shape
        target_shape_set = {ode_inputs[tgt]['shape'] for tgt in targets}
        if len(target_shape_set) == 1:
            shape = next(iter(target_shape_set))
            if not static_target:
                if len(shape) == 1:
                    shape = (1,)
                else:
                    shape = shape[1:]
        elif len(target_shape_set) == 0:
            raise ValueError(f'Unable to automatically assign a shape to {name}.\n'
                             'Targets for this variable either do not exist or have no shape set.\n'
                             'The shape for this variable must be set explicitly via the '
                             '`shape=<tuple>` argument.')
        else:
            raise ValueError(f'Unable to automatically assign a shape to {name} based on targets. '
                             f'Targets have multiple shapes assigned: {target_shape_set}. '
                             f'Change targets such that all have common shapes.')
    else:
        shape = user_shape

    return targets, shape, units, static_target


def get_source_metadata(ode, src, user_units=_unspecified, user_shape=_unspecified):
    """
    Return the units and shape of output src in the given ODE.

    If the targets of the state is _unspecified, and the state name is a top level input name
    in the ODE, then the state values are automatically connected to that top-level input.
    If _unspecified and not a top-level input of the ODE, no connection is made.
    If targets is explicitly None, then no connection is made.
    Otherwise, if the user specified some other string or sequence of strings as targets, then
    those are returned.

    Parameters
    ----------
    ode : openmdao.core.System or dict
        The OpenMDAO system which serves as the ODE for dymos, or a dictionary of promoted output names and
        corresponding metadata from get_promoted_vars.  If given as a system, it should already have
        had its setup and configure methods called.
    src : str
        The relative path in the ODE to the source variable whose metadata is requested.
    user_units : str or None or Sequence or _unspecified
        Units for the variable as given by the user.
    user_shape : str or None or Sequence or _unspecified
        Shape for the variable as given by the user.

    Returns
    -------
    dict
        A dictionary containing the metadata for the source. This consists of shape, units, and tags.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    meta = {}
    ode_outputs = ode if isinstance(ode, dict) else get_promoted_vars(ode, iotypes='output')

    if src not in ode_outputs:
        raise ValueError(f"Unable to find the source '{src}' in the ODE.")

    if user_units in {None, _unspecified}:
        meta['units'] = ode_outputs[src]['units']
    else:
        meta['units'] = user_units

    if user_shape in {None, _unspecified}:
        ode_shape = ode_outputs[src]['shape']
        meta['shape'] = (1,) if len(ode_shape) == 1 else ode_shape[1:]
    else:
        meta['shape'] = user_shape

    meta['tags'] = ode_outputs[src]['tags']

    return meta
