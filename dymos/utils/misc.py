import sys

import numpy as np

_unspecified = object()


def get_rate_units(units, time_units, deriv=1):
    """
    Return a string for rate units given units for the variable
    and time units.

    Parameters
    ----------
    units : str
        Units of a given variable.
    time_units : str
        Time units.
    deriv : int
        If 1, provide the units of the first derivative.  If 2,
        provide the units of the second derivative.

    Returns
    -------
    str
        Corresponding rate units for the given variable.

    """
    if deriv not in (1, 2):
        raise ValueError('deriv argument must be 1 or 2.')

    tu = time_units if deriv == 1 else '{0}**2'.format(time_units)

    if units is not None and time_units is not None:
        rate_units = '{0}/{1}'.format(units, tu)
    elif units is not None:
        rate_units = units
    elif time_units is not None:
        rate_units = '1.0/{0}'.format(tu)
    else:
        rate_units = None
    return rate_units


def get_targets(ode, name, user_targets):
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
    ode_inputs = [opts['prom_name'] for (k, opts) in ode.get_io_metadata(iotypes=('input',)).items()]

    if user_targets is _unspecified:
        if name in ode_inputs:
            targets = [name]
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


class CoerceDesvar(object):
    """
    Check the desvar options for the appropriate shape and resize
    accordingly with options.
    """
    def __init__(self, num_input_nodes, desvar_indices, options):
        self.num_input_nodes = num_input_nodes
        self.desvar_indices = desvar_indices
        self.options = options

    def __call__(self, option):
        """
        Test that the given opption has a shape that is compliant with the number of input
        nodes for the design variable.

        Parameters
        ----------
        option : str
            The name of the option whose value(s) are desired.

        Returns
        -------
        The value of the desvar option

        Raises
        ------
        ValueError
            If the number of values in the option is not compliant with the number of input
            nodes for the design variable.

        """
        val = self.options[option]
        if val is None or np.isscalar(val):
            return val
        else:
            if len(val) != self.num_input_nodes:
                raise ValueError('array-valued option {0} must have length '
                                 'num_input_nodes ({1})'.format(option, val))
            return val[self.desvar_indices]
