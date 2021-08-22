import openmdao.api as om
from dymos.utils.misc import _unspecified
from ..phase.options import StateOptionsDictionary
from .misc import get_target_metadata, get_rate_units, get_source_metadata


def classify_var(var, state_options, parameter_options, control_options, polynomial_control_options):
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

    Returns
    -------
    str
        The classification of the given variable, which is one of
        'time', 'time_phase', 'state', 'input_control', 'indep_control', 'control_rate',
        'control_rate2', 'input_polynomial_control', 'indep_polynomial_control',
        'polynomial_control_rate', 'polynomial_control_rate2', 'parameter',
        or 'ode'.
    """
    if var == 'time':
        return 'time'
    elif var == 'time_phase':
        return 'time_phase'
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
    elif var.endswith('_rate') and var[:-5] in control_options:
        return 'control_rate'
    elif var.endswith('_rate2') and var[:-6] in control_options:
        return 'control_rate2'
    elif var.endswith('_rate') and var[:-5] in polynomial_control_options:
        return 'polynomial_control_rate'
    elif var.endswith('_rate2') and var[:-6] in polynomial_control_options:
        return 'polynomial_control_rate2'
    else:
        return 'ode'


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
    ode : om.System
        The OpenMDAO system which serves as the ODE for dymos.  This system should already have
        had its setup and configure methods called.
    name : str
        The name of the state variable whose targets are desired.
    user_targets : str or None or Sequence or _unspecified
        Targets for the variable as given by the user.

    Returns
    -------
    list
        The target inputs of the state variable in the ODE, as a list.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    ode_inputs = {opts['prom_name']: opts for (k, opts) in
                  ode.get_io_metadata(iotypes=('input',), get_remote=True).items()}

    if user_targets is _unspecified:
        if name in ode_inputs:
            targets = [name]
        elif control_rates and f'{name}_rate' in ode_inputs:
            targets = [f'{name}_rate']
        elif control_rates and f'{name}_rate2' in ode_inputs:
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

    return targets


def get_state_target_metadata(ode, name, targets=_unspecified, user_units=_unspecified,
                              user_shape=_unspecified):
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
    ode : om.System
        The OpenMDAO system which serves as the ODE for dymos.  This system should already have
        had its setup and configure methods called.
    name : str
        The name of the state variable whose targets are desired.
    targets : Sequence
        Targets for the variable (assumes get_targets has already been run).
    user_units : str or None or Sequence or _unspecified
        Units for the variable as given by the user.
    user_shape : str or None or Sequence or _unspecified
        Shape for the variable as given by the user.

    Returns
    -------
    tuple
        The shape of the variable.  If not specified, shape is taken from the ODE targets.
    str
        The units of the variable.  If not specified, units are taken from the ODE targets.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    rate_src = False
    ode_inputs = {opts['prom_name']: opts for (k, opts) in
                  ode.get_io_metadata(iotypes=('input', 'output'), get_remote=True).items()}

    if user_units is _unspecified:
        target_units_set = {ode_inputs[tgt]['units'] for tgt in targets}
        if len(target_units_set) == 1:
            units = next(iter(target_units_set))
            if rate_src:
                units = f"{units}*s"
        else:
            raise ValueError(f'Unable to automatically assign units to {name}. '
                             f'Targets have multiple units: {target_units_set}. '
                             f'Either promote targets and use set_input_defaults to assign common '
                             f'units, or explicitly provide them to {name}.')
    else:
        units = user_units

    if user_shape in {None, _unspecified}:
        target_shape_set = {ode_inputs[tgt]['shape'] for tgt in targets}
        if len(target_shape_set) == 1:
            shape = next(iter(target_shape_set))
            if len(shape) == 1:
                shape = (1,)
            else:
                shape = shape[1:]
        elif len(target_shape_set) == 0:
            raise ValueError(f'Unable to automatically assign a shape to {name}. '
                             'Independent controls need to declare a shape.')
        else:
            raise ValueError(f'Unable to automatically assign a shape to {name} based on targets. '
                             f'Targets have multiple shapes assigned: {target_shape_set}. '
                             f'Change targets such that all have common shapes.')
    else:
        shape = user_shape

    return shape, units


def configure_controls_introspection(control_options, ode):
    """
    Modify control options in-place using introspection of the user-provided ODE.

    Parameters
    ----------
    control_options : dict of {str: ControlOptionsDictionary
        A dictionary keyed by control name containing the options for all controls to be applied
        to the ODE.
    ode : om.System
        An instantiated System that serves as the ODE to which the controls should be applied.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the control or one of its rates are connected to a variable that is tagged as static
        within the ODE.
    """
    for name, options in control_options.items():
        options['targets'] = get_targets(ode, name, options['targets'], control_rates=True)

        shape, units, static_target = get_target_metadata(ode, name=name,
                                                          user_targets=options['targets'],
                                                          user_units=options['units'],
                                                          user_shape=options['shape'],
                                                          control_rate=True)

        options['units'] = units
        options['shape'] = shape

        if static_target:
            raise ValueError(f"Control '{name}' cannot be connected to its targets because one "
                             f"or more targets are tagged with 'dymos.static_target'.")

        # Now check rate targets
        _, _, static_target = get_target_metadata(ode, name=name,
                                                  user_targets=options['rate_targets'],
                                                  user_units=options['units'],
                                                  user_shape=options['shape'],
                                                  control_rate=True)
        if static_target:
            raise ValueError(f"Control rate of '{name}' cannot be connected to its targets "
                             f"because one or more targets are tagged with 'dymos.static_target'.")

        # Now check rate2 targets
        _, _, static_target = get_target_metadata(ode, name=name,
                                                  user_targets=options['rate2_targets'],
                                                  user_units=options['units'],
                                                  user_shape=options['shape'],
                                                  control_rate=True)
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
        to the ODE.
    ode : om.System
        An instantiated System that serves as the ODE to which the paremeters should be applied.

    Returns
    -------
    None
    """
    for name, options in parameter_options.items():
        options['targets'] = get_targets(ode, name, options['targets'])

        shape, units, static_target = get_target_metadata(ode, name=name,
                                                          user_targets=options['targets'],
                                                          user_units=options['units'],
                                                          user_shape=options['shape'])

        options['units'] = units
        options['shape'] = shape


def configure_time_introspection(time_options, ode):
    """
    Modify time options in-place using introspection of the user-provided ODE.

    Parameters
    ----------
    time_options : dict of {str: TimeOptionsDictionary
        A dictionary keyed by control name containing the options for all controls to be applied
        to the ODE.
    ode : om.System
        An instantiated System that serves as the ODE to which the controls should be applied.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If time or time_phase are connected to a variable that is tagged as static
        within the ODE.
    """
    # time
    time_options['targets'] = get_targets(ode, 'time', time_options['targets'])

    _, units, static_target = get_target_metadata(ode, name='time',
                                                  user_targets=time_options['targets'],
                                                  user_units=time_options['units'],
                                                  user_shape=(1,))

    time_options['units'] = units

    if static_target:
        raise ValueError(f"'time' cannot be connected to its targets because one "
                         f"or more targets are tagged with 'dymos.static_target'.")

    # time_phase
    time_options['time_phase_targets'] = get_targets(ode, 'time_phase', time_options['time_phase_targets'])

    _, _, static_target = get_target_metadata(ode, name='time_phase',
                                              user_targets=time_options['time_phase_targets'],
                                              user_units=time_options['units'],
                                              user_shape=(1,))

    if static_target:
        raise ValueError(f"'time_phase' cannot be connected to its targets because one "
                         f"or more targets are tagged with 'dymos.static_target'.")


def configure_states_introspection(state_options, time_options, control_options, parameter_options,
                                   polynomial_control_options, ode):
    """
    Modifies state options in-place, automatically determining 'targets', 'units', and 'shape'
    if necessary.

    The precedence rules for the state shape and units are as follows:
    1. If the user has specified units and shape in the state options, use those.
    2a. If the user has not specified shape, and targets exist, then pull the shape from the targets.
    2b. If the user has not specified shape and no targets exist, then pull the shape from the rate source.
    2c. If shape cannot be inferred, assume (1,)
    3a. If the user has not specified units, first try to pull units from a target
    3b. If there are no targets, pull units from the rate source and multiply by time units.

    Parameters
    ----------
    state_name : str
        The name of the state variable of interest.
    options : OptionsDictionary
        The options dictionary for the state variable of interest.
    phase : dymos.Phase
        The phase associated with the transcription.

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

    Returns
    -------
    None
    """
    time_units = time_options['units']

    for state_name, options in state_options.items():
        user_targets = options['targets']
        user_units = options['units']
        user_shape = options['shape']

        need_units = user_units is _unspecified
        need_shape = user_shape in {None, _unspecified}

        # Automatically determine targets of state if left _unspecified
        if user_targets is _unspecified:
            options['targets'] = get_targets(ode, state_name, user_targets)

        # 1. No introspection necessary
        if not(need_shape or need_units):
            return

        # 2. Attempt target introspection
        if options['targets']:
            try:
                tgt_shape, tgt_units = get_state_target_metadata(ode, state_name, options['targets'],
                                                                 options['units'], options['shape'])
                options['shape'] = tgt_shape
                options['units'] = tgt_units
                return
            except ValueError:
                pass

        # 3. Attempt rate-source introspection
        rate_src = options['rate_source']
        rate_src_type = classify_var(rate_src, state_options, parameter_options, control_options,
                                     polynomial_control_options)

        if rate_src_type in ['time', 'time_phase']:
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
            rate_src_shape, rate_src_units = get_source_metadata(ode,
                                                                 src=rate_src,
                                                                 user_units=options['units'],
                                                                 user_shape=options['shape'])
        else:
            rate_src_shape = (1,)
            rate_src_units = None

        if need_shape:
            options['shape'] = rate_src_shape

        if need_units:
            options['units'] = time_units if rate_src_units is None else f'{rate_src_units}*{time_units}'

def configure_states_discovery(state_options, ode):
    """
    Searches phase output metadata for any declared states and adds them.

    Parameters
    ----------
    phase : dymos.Phase
        The phase object to which this transcription instance applies.
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
                state = tag.split(':')[-1]
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
                tagged_state_units = tag.split(':')[-1]
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
