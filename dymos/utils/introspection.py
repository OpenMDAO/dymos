from dymos.utils.misc import _unspecified
from .misc import get_target_metadata


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

def configure_control_introspection(control_options, ode):
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
    # Interrogate shapes and units and static/dynamic behavior
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
