import fnmatch
from numbers import Number

import openmdao.api as om
import numpy as np
from openmdao.utils.units import simplify_unit
from openmdao.utils.general_utils import ensure_compatible
from dymos.utils.misc import _unspecified, is_unspecified, is_none_or_unspecified
from ..phase.options import StateOptionsDictionary, TimeseriesOutputOptionsDictionary
from .misc import get_rate_units


def classify_var(var, time_options, state_options, parameter_options, control_options,
                 timeseries_options=None):
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
    timeseries_options : {str: OptionsDictionary}
        For each timeseries, a dictionary of its options, keyed by name.

    Returns
    -------
    str
        The classification of the given variable, which is one of
        't', 't_phase', 'state', 'control', 'control_rate',
        'control_rate2', 'parameter', or 'ode'.
    """
    time_name = time_options['name']
    if var == time_name:
        return 't'
    elif var == f'{time_name}_phase':
        return 't_phase'
    elif var == 'time_phase':
        om.issue_warning('time_phase is deprecated. Please use `t_phase` to obtain the change in the integration '
                         'variable within the current phase.', category=om.OMDeprecationWarning)
        return 't_phase'
    elif var.startswith('initial_states:'):
        return 'state'
    elif var.startswith('final_states:'):
        return 'state'
    elif var in state_options:
        return 'state'
    elif var in control_options:
        return 'control'
    elif var in parameter_options:
        return 'parameter'
    elif var.endswith('_rate'):
        if var[:-5] in control_options:
            return 'control_rate'
    elif var.endswith('_rate2'):
        if var[:-6] in control_options:
            return 'control_rate2'
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


def get_targets(ode, name, user_targets):
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
    if is_unspecified(user_targets):
        if name in ode_inputs:
            return [name]
    elif user_targets:
        if isinstance(user_targets, str):
            return [user_targets]
        else:
            return user_targets
    return []


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
    ode_inputs = get_promoted_vars(ode, iotypes='input', metadata_keys=['shape', 'units', 'val', 'tags'])
    for name, options in control_options.items():
        targets = _get_targets_metadata(ode_inputs, name=name, user_targets=options['targets'])

        options['targets'] = list(targets.keys())
        if targets:
            if is_unspecified(options['units']):
                options['units'] = _get_common_metadata(targets, metadata_key='units')

            if is_none_or_unspecified(options['shape']):
                shape = _get_common_metadata(targets, metadata_key='shape')
                if len(shape) == 1:
                    options['shape'] = (1,)
                else:
                    options['shape'] = shape[1:]

            if any(['dymos.static_target' in meta['tags'] for meta in targets.values()]):
                raise ValueError(f"Control '{name}' cannot be connected to its targets because one "
                                 f"or more targets are tagged with 'dymos.static_target'.")

        # Now check rate targets
        rate_targets = _get_targets_metadata(ode_inputs, name=f'{name}_rate',
                                             user_targets=options['rate_targets'])

        options['rate_targets'] = list(rate_targets.keys())
        if rate_targets:
            if is_unspecified(options['units']):
                rate_target_units = _get_common_metadata(rate_targets, metadata_key='units')
                options['units'] = time_units if rate_target_units is None else \
                    simplify_unit(f'{rate_target_units}*{time_units}')

            if is_none_or_unspecified(options['shape']):
                shape = _get_common_metadata(rate_targets, metadata_key='shape')
                if len(shape) == 1:
                    options['shape'] = (1,)
                else:
                    options['shape'] = shape[1:]

            if any(['dymos.static_target' in meta['tags'] for meta in rate_targets.values()]):
                raise ValueError(f"Control rate of '{name}' cannot be connected to its targets because one "
                                 f"or more targets are tagged with 'dymos.static_target'.")

        # Now check rate2 targets
        rate2_targets = _get_targets_metadata(ode_inputs, name=f'{name}_rate2',
                                              user_targets=options['rate2_targets'])

        options['rate2_targets'] = list(rate2_targets.keys())
        if rate2_targets:
            if is_unspecified(options['units']):
                rate2_target_units = _get_common_metadata(rate_targets, metadata_key='units')
                options['units'] = f'{time_units**2}' if rate2_target_units is None \
                    else simplify_unit(f'{rate2_target_units}*{time_units}**2')

            if is_none_or_unspecified(options['shape']):
                shape = _get_common_metadata(rate2_targets, metadata_key='shape')
                if len(shape) == 1:
                    options['shape'] = (1,)
                else:
                    options['shape'] = shape[1:]

            if any(['dymos.static_target' in meta['tags'] for meta in rate2_targets.values()]):
                raise ValueError(f"Control rate2 of '{name}' cannot be connected to its targets because one "
                                 f"or more targets are tagged with 'dymos.static_target'.")


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
    for name, options in parameter_options.items():
        try:
            targets = _get_targets_metadata(ode, name=name, user_targets=options['targets'])
        except ValueError as e:
            raise ValueError(f'Parameter `{name}` has invalid target(s).\n{str(e)}') from e

        options['targets'] = list(targets.keys())

        static_tagged_targets = {tgt for tgt, meta in targets.items() if 'dymos.static_target' in meta['tags']}

        # This is a bit of a hack. Any target with a shape of (1,) is unambiguously static.
        # We may want to consider forcing users to tag these as static for dymos 2.0.0
        shape_1_targets = {tgt for tgt, meta in targets.items() if meta['shape'] == (1,)}
        if is_unspecified(options['static_targets']):
            options['static_targets'] = static_tagged_targets.union(shape_1_targets)
        elif options['static_targets']:
            options['static_targets'] = options['targets'].copy()
        else:
            options['static_targets'] = []

        if static_tagged_targets and not options['static_targets']:
            raise ValueError(f"Parameter `{name}` has invalid target(s).\n"
                             f"User has specified 'static_target = False' for parameter `{name}`,\nbut one or more "
                             f"targets is tagged with 'dymos.static_target':\n{static_tagged_targets}")

        if is_unspecified(options['units']):
            options['units'] = _get_common_metadata(targets, metadata_key='units')

        # Check that all targets have the same shape.
        tgt_shapes = {}
        # First find the shapes of the static targets
        for tgt, meta in targets.items():
            if tgt in options['static_targets']:
                tgt_shapes[tgt] = meta['shape']
            else:
                if len(meta['shape']) == 1:
                    tgt_shapes[tgt] = (1,)
                else:
                    tgt_shapes[tgt] = meta['shape'][1:]
        # Check that they're unique
        if len(set(tgt_shapes.values())) > 1:
            raise RuntimeError(f'Invalid targets for parameter `{name}`.\n'
                               f'Targets have multiple shapes.\n'
                               f'{tgt_shapes}')
        elif len(set(tgt_shapes.values())) == 1:
            introspected_shape = next(iter(set(tgt_shapes.values())))
        else:
            introspected_shape = None

        if is_none_or_unspecified(options['shape']):
            if isinstance(options['val'], Number):
                options['shape'] = introspected_shape
            else:
                options['shape'] = np.asarray(options['val']).shape
        else:
            if introspected_shape is not None and options['shape'] != introspected_shape:
                raise RuntimeError(f'Shape provided to parameter `{name}` differs from its targets.\n'
                                   f'Given shape: {options["shape"]}\n'
                                   f'Target shapes:\n'
                                   f'{tgt_shapes}')

        options['val'], options['shape'] = ensure_compatible(name, options['val'], options['shape'])


def configure_time_introspection(time_options, ode):
    """
    Modify time options in-place using introspection of the user-provided ODE.

    Parameters
    ----------
    time_options : dict of {str: TimeOptionsDictionary
        A dictionary keyed by control name containing the options for all controls to be applied
        to the ODE. Options for 'targets', 't_phase_targets', and 'units' are modified in-place.
    ode : om.System
        An instantiated System that serves as the ODE to which the controls should be applied.

    Raises
    ------
    ValueError
        If time or time_phase are connected to a variable that is tagged as static
        within the ODE.
    """
    ode_inputs = get_promoted_vars(ode, 'input', metadata_keys=['shape', 'val', 'units', 'tags'])
    time_name = time_options['name']
    t_phase_name = f'{time_name}_phase'

    targets = _get_targets_metadata(ode_inputs, name=time_name, user_targets=time_options['targets'])

    time_options['targets'] = targets

    if is_unspecified(time_options['units']):
        time_options['units'] = _get_common_metadata(targets, 'units')

    if any(['dymos.static_target' in meta['tags'] for meta in targets.values()]):
        raise ValueError(f"The integration variable {time_name} cannot be connected to its targets because one "
                         f"or more targets are tagged with 'dymos.static_target'.")

    # t_phase
    targets = _get_targets_metadata(ode_inputs, name=t_phase_name, user_targets=time_options['time_phase_targets'])

    time_options['time_phase_targets'] = targets

    if any(['dymos.static_target' in meta['tags'] for meta in targets.values()]):
        raise ValueError("'t_phase' cannot be connected to its targets because one "
                         "or more targets are tagged with 'dymos.static_target'.")


def configure_states_introspection(state_options, time_options, control_options, parameter_options,
                                   ode):
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
    ode : System
        The OpenMDAO system which provides the state rates as outputs.
    """
    time_units = time_options['units']
    ode_inputs = get_promoted_vars(ode, 'input', metadata_keys=['units', 'shape', 'val', 'tags'])
    ode_outputs = get_promoted_vars(ode, 'output', metadata_keys=['units', 'shape', 'val', 'tags'])

    for state_name, options in state_options.items():
        # Automatically determine targets of state if left _unspecified
        targets = _get_targets_metadata(ode_inputs, state_name,
                                        user_targets=options['targets'])

        options['targets'] = list(targets.keys())
        if targets:
            if is_unspecified(options['units']):
                options['units'] = _get_common_metadata(targets, metadata_key='units')

            if is_none_or_unspecified(options['shape']):
                shape = _get_common_metadata(targets, metadata_key='shape')
                if len(shape) == 1:
                    options['shape'] = (1,)
                else:
                    options['shape'] = shape[1:]

            if any(['dymos.static_target' in meta['tags'] for meta in targets.values()]):
                raise ValueError(f"State '{state_name}' cannot be connected to its targets because one "
                                 f"or more targets are tagged with 'dymos.static_target'.")

        # 3. Attempt rate-source introspection
        rate_src = options['rate_source']
        rate_src_type = classify_var(rate_src, time_options, state_options, parameter_options, control_options)

        if rate_src_type in {'t', 't_phase'}:
            rate_src_units = time_options['units']
            rate_src_shape = (1,)
        elif rate_src_type == 'state':
            rate_src_units = state_options[rate_src]['units']
            rate_src_shape = state_options[rate_src]['shape']
        elif rate_src_type == 'control':
            rate_src_units = control_options[rate_src]['units']
            rate_src_shape = control_options[rate_src]['shape']
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
        elif rate_src_type == 'ode':
            meta = get_source_metadata(ode_outputs, src=rate_src, user_units=options['units'], user_shape=options['shape'])
            rate_src_shape = meta['shape']
            rate_src_units = meta['units']
        else:
            rate_src_shape = (1,)
            rate_src_units = None

        if is_none_or_unspecified(options['shape']):
            options['shape'] = rate_src_shape

        if is_unspecified(options['units']):
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
                    options['units'] = simplify_unit(f'{rate_src_units}*{time_units}')


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

        if is_none_or_unspecified(options['shape']):
            options['shape'] = src_shape

        if is_unspecified(options['units']):
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

            try:
                output_meta = transcription._get_timeseries_var_source(output_options['name'],
                                                                       output_options['output_name'],
                                                                       phase=phase)
            except ValueError:
                not_found.add(output_name)
                raise

            output_options['src'] = output_meta['src']
            output_options['src_idxs'] = output_meta['src_idxs']

            if is_none_or_unspecified(output_options['shape']):
                output_options['shape'] = output_meta['shape']

            if is_none_or_unspecified(output_options['units']):
                output_options['units'] = output_meta['units']

        if not_found:
            sorted_list = ', '.join(sorted([ts_opts['outputs'][output_name]['name'] for output_name in not_found]))
            om.issue_warning(f'{phase.pathname}: The following timeseries outputs were requested but not found in the '
                             f'ODE: {sorted_list}')

        for s in not_found:
            ts_opts['outputs'].pop(s)


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


def _configure_boundary_balance_introspection(phase):
    """
    Modify duration balance options in-place using introspection of the phase and its ODE.

    Parameters
    ----------
    phase : Phase
        The phase object whose boundary and path constraints are to be introspected.
    """
    time_units = phase.time_options['units']

    for param_name, options in phase.boundary_balance_options.items():
        resid_name = options['name']
        resid_units = options.get('eq_units', _unspecified)
        resid_shape = options.get('shape', None)

        # Determine the path to the variable which we will be constraining
        resid_type = phase.classify_var(resid_name)

        if param_name == 't_initial':
            param_units = phase.time_options['units']
            param_bounds = phase.time_options['initial_bounds']
        elif param_name == 't_duration':
            param_units = phase.time_options['units']
            param_bounds = phase.time_options['duration_bounds']
        elif param_name in phase.parameter_options:
            param_units = phase.parameter_options[param_name]['units']
            param_bounds = tuple(phase.parameter_options[param_name][k] for k in ('lower', 'upper'))
        elif param_name.startswith('initial_states:'):
            state_name = ':'.join(param_name.split(':')[1:])
            param_units = phase.state_options[state_name]['units']
            param_bounds = phase.state_options[state_name]['initial_bounds']
        elif param_name.startswith('final_states:'):
            state_name = ':'.join(param_name.split(':')[1:])
            param_units = phase.state_options[state_name]['units']
            param_bounds = phase.state_options[state_name]['final_bounds']
        else:
            raise ValueError(f'{phase.msginfo}: For boundary balance, param must be one of t_initial, t_duration, '
                             'a parameter in the phase, initial_states:{name}, or final_states:{name}')

        if is_unspecified(options.get('units', _unspecified)):
            options['units'] = param_units

        if is_unspecified(options.get('lower', _unspecified)):
            if param_bounds is None:
                options['lower'] = None
            else:
                options['lower'] = param_bounds[0]

        if is_unspecified(options.get('upper', _unspecified)):
            if param_bounds is None:
                options['upper'] = None
            else:
                options['upper'] = param_bounds[1]

        if is_unspecified(resid_units):
            if resid_type in ['t', 't_phase']:
                options['eq_units'] = time_units
            elif resid_type == 'state':
                if resid_name.startswith('initial_states:') or resid_name.startswith('final_states:'):
                    state_name = ':'.join(resid_name.split(':')[1:])
                else:
                    state_name = resid_name
                options['eq_units'] = phase.state_options[state_name]['units']
            elif resid_type == 'parameter':
                options['eq_units'] = phase.parameter[resid_name]['units']
            elif resid_type == 'control':
                options['eq_units'] = phase.control_options[resid_name]['units']
            elif resid_type == 'control_rate':
                control_units = phase.state_options[resid_name]['units']
                options['eq_units'] = get_rate_units(control_units, time_units, deriv=1)
            elif resid_type == 'control_rate2':
                options['eq_units'] = get_rate_units(control_units, time_units, deriv=2)
            elif resid_type == 'ode':
                # Failed to find variable, assume it is in the ODE. This requires introspection.
                ode = phase.options['transcription']._get_ode(phase)
                meta = get_source_metadata(ode, src=resid_name, user_units=resid_units, user_shape=resid_shape)
                options['eq_units'] = meta['units']
            else:
                raise ValueError(f'{phase.msginfo}: Unable to find boundary balance name {resid_name}')


def _configure_constraint_introspection(phase):
    """
    Modify constraint options in-place using introspection of the phase and its ODE.

    Parameters
    ----------
    phase : Phase
        The phase object whose boundary and path constraints are to be introspected.
    """
    has_boundary_ode = phase.options['transcription']._has_boundary_ode
    has_initial_final_states = phase.options['transcription']._has_boundary_ode or \
        phase.options['transcription']._has_initial_final_states

    for constraint_type, constraints in [('initial', phase._initial_boundary_constraints),
                                         ('final', phase._final_boundary_constraints),
                                         ('path', phase._path_constraints)]:

        time_units = phase.time_options['units']
        time_name = phase.time_options['name']

        for con in constraints:
            # Determine the path to the variable which we will be constraining
            var = con['name']
            var_type = phase.classify_var(var)

            if var != con['constraint_name'] is not None and var_type != 'ode':
                om.issue_warning(f"Option 'constraint_name' on {constraint_type} constraint {var} is only "
                                 f"valid for ODE outputs. The option is being ignored.", om.UnusedOptionWarning)

            if var_type == 't':
                con['shape'] = (1,)
                con['units'] = time_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.{time_name}'

            elif var_type == 't_phase':
                con['shape'] = (1,)
                con['units'] = time_units if con['units'] is None else con['units']
                con['constraint_path'] = f'timeseries.{time_name}_phase'

            elif var_type == 'state':
                prefix = 'states:' if phase.timeseries_options['use_prefix'] else ''
                state_shape = phase.state_options[var]['shape']
                state_units = phase.state_options[var]['units']
                con['shape'] = state_shape
                con['units'] = state_units if con['units'] is None else con['units']
                if has_initial_final_states and constraint_type == 'initial':
                    con['constraint_path'] = f'initial_states:{var}'
                elif has_initial_final_states and constraint_type == 'final':
                    con['constraint_path'] = f'final_states:{var}'
                else:
                    con['constraint_path'] = f'timeseries.{prefix}{var}'

            elif var_type == 'parameter':
                param_shape = phase.parameter_options[var]['shape']
                param_units = phase.parameter_options[var]['units']
                con['shape'] = param_shape
                con['units'] = param_units if con['units'] is None else con['units']
                con['constraint_path'] = f'parameter_vals:{var}'

            elif var_type == 'control':
                prefix = 'controls:' if phase.timeseries_options['use_prefix'] else ''
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                con['shape'] = control_shape
                con['units'] = control_units if con['units'] is None else con['units']
                if has_boundary_ode and constraint_type in ('initial', 'final'):
                    con['constraint_path'] = f'boundary_vals.{var}'
                else:
                    con['constraint_path'] = f'timeseries.{prefix}{var}'

            elif var_type == 'control_rate':
                prefix = 'control_rates:' if phase.timeseries_options['use_prefix'] else ''
                control_name = var[:-5]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                con['shape'] = control_shape
                con['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con['units'] is None else con['units']
                if has_boundary_ode and constraint_type in ('initial', 'final'):
                    con['constraint_path'] = f'boundary_vals.{var}'
                else:
                    con['constraint_path'] = f'timeseries.{prefix}{var}'

            elif var_type == 'control_rate2':
                prefix = 'control_rates:' if phase.timeseries_options['use_prefix'] else ''
                control_name = var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                con['shape'] = control_shape
                con['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con['units'] is None else con['units']
                if has_boundary_ode and constraint_type in ('initial', 'final'):
                    con['constraint_path'] = f'boundary_vals.{var}'
                else:
                    con['constraint_path'] = f'timeseries.{prefix}{var}'

            else:
                # Failed to find variable, assume it is in the ODE. This requires introspection.
                ode = phase.options['transcription']._get_ode(phase)

                meta = get_source_metadata(ode, src=var, user_units=con['units'], user_shape=con['shape'])

                con['shape'] = meta['shape']
                con['units'] = meta['units']

                if has_boundary_ode and constraint_type in ('initial', 'final'):
                    con['constraint_path'] = f'boundary_vals.{var}'
                else:
                    con['constraint_path'] = f'timeseries.{con["constraint_name"]}'


def _get_targets_metadata(ode, name, user_targets=_unspecified):
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

    Returns
    -------
    targets : dict
        A dictionary of target inputs in the ODE and metadata associated with each target.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    ode_inputs = ode if isinstance(ode, dict) else get_promoted_vars(ode,
                                                                     iotypes='input',
                                                                     metadata_keys=['shape', 'units',
                                                                                    'val', 'tags'])

    targets = {t: {} for t in get_targets(ode_inputs, name, user_targets=user_targets)}

    for tgt in targets:
        if tgt not in ode_inputs:
            raise ValueError(f"No such ODE input: '{tgt}'.")

    for tgt, options in targets.items():
        options['units'] = ode_inputs[tgt]['units']
        options['shape'] = ode_inputs[tgt]['shape']
        options['val'] = ode_inputs[tgt]['val']
        options['tags'] = ode_inputs[tgt]['tags']

    return targets


def _get_common_metadata(targets, metadata_key):
    """
    Given a dictionary containing targets and their metadata, return the value associated
    with the given metadata key if that value is common to all targets, otherwise raise an
    Exception.

    Parameters
    ----------
    targets : dict
        A dictionary of targets and their metadata which must include the desired metadata key.
    metadata_key : str
        The metadata key desired.

    Returns
    -------
    object
        The common metadata value shared by all targets.

    Raises
    ------
    ValueError
        ValueError is raised if the targets do not all have the same metadata value.
    """
    meta_set = {meta[metadata_key] for meta in targets.values()}

    if len(meta_set) == 1:
        return next(iter(meta_set))
    elif len(meta_set) == 0:
        raise RuntimeError(f'Unable to automatically assign {metadata_key} based on targets. \n'
                           f'No targets were found.')
    else:
        err_dict = {tgt: meta[metadata_key] for tgt, meta in targets.items()}
        raise RuntimeError(f'Unable to automatically assign {metadata_key} based on targets.\n'
                           f'Targets have multiple {metadata_key} assigned:\n{err_dict}.\n'
                           f'Either promote targets and use set_input_defaults to assign common\n'
                           f'{metadata_key}, or explicitly provide {metadata_key} to the variable.')


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

    if is_none_or_unspecified(user_units):
        meta['units'] = ode_outputs[src]['units']
    else:
        meta['units'] = user_units

    if is_none_or_unspecified(user_shape):
        ode_shape = ode_outputs[src]['shape']
        meta['shape'] = (1,) if len(ode_shape) == 1 else ode_shape[1:]
    else:
        meta['shape'] = user_shape

    meta['tags'] = ode_outputs[src]['tags']

    return meta
