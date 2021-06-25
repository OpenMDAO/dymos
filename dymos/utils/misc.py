import numpy as np


class _ReprClass(object):
    """
    Class for defining objects with a simple constant string __repr__.

    This is useful for constants used in arg lists when you want them to appear in
    automatically generated source documentation as a certain string instead of python's
    default representation.
    """

    def __init__(self, repr_string):
        """
        Initialize the __repr__ string.

        Parameters
        ----------
        repr_string : str
            The string to be returned by __repr__
        """
        self._repr_string = repr_string

    def __repr__(self):
        """
        Return our _repr_string.

        Returns
        -------
        str
            Whatever string we were initialized with.
        """
        return self._repr_string

    def __str__(self):
        return self._repr_string


# unique object to check if default is given (when None is an allowed value)
_unspecified = _ReprClass("unspecified")


def get_rate_units(units, time_units, deriv=1):
    """
    Return a string for rate units given units for the variable and time units.

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
    ode : om.System
        The OpenMDAO system which serves as the ODE for dymos.  This system should already have
        had its setup and configure methods called.
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

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    rate_src = False
    ode_inputs = {opts['prom_name']: opts for (k, opts) in
                  ode.get_io_metadata(iotypes=('input',), get_remote=True).items()}

    if user_targets is _unspecified:
        if name in ode_inputs:
            targets = [name]
        elif control_rate and f'{name}_rate' in ode_inputs:
            targets = [f'{name}_rate']
            rate_src = True
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
            if rate_src:
                units = f"{units}*s"
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


def get_source_metadata(ode, src, user_units, user_shape):
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
    src : str
        The relative path in the ODE to the source variable whose metadata is requested.
    user_units : str or None or Sequence or _unspecified
        Units for the variable as given by the user.
    user_shape : str or None or Sequence or _unspecified
        Shape for the variable as given by the user.

    Returns
    -------
    shape : tuple
        The shape of the variable.  If not specified, shape is taken from the ODE targets.
    units : str
        The units of the variable.  If not specified, units are taken from the ODE targets.

    Notes
    -----
    This method requires that the ODE has run its setup and configure methods.  Thus,
    this method should be called from configure of some parent Group, and the ODE should
    be a system within that Group.
    """
    ode_outputs = {opts['prom_name']: opts for (k, opts) in
                   ode.get_io_metadata(iotypes=('output',), get_remote=True).items()}

    if src not in ode_outputs:
        raise ValueError(f'Unable to find the source {src} in the ODE at {ode.pathname}.')

    if user_units in {None, _unspecified}:
        units = ode_outputs[src]['units']
    else:
        units = user_units

    if user_shape in {None, _unspecified}:
        ode_shape = ode_outputs[src]['shape']
        shape = (1,) if len(ode_shape) == 1 else ode_shape[1:]
    else:
        shape = user_shape

    return shape, units


def reshape_val(val, shape, num_input_nodes):
    """
    Return the given value reshaped to (num_input_nodes,) + shape.

    If the value is scalar or a size-1 array, return that value multiplied by
    np.ones((num_input_nodes,) + shape).  If the value's shape is shape, then
    repeat those values along a new first dimension.  Otherwise, reshape it to
    the correct shape and return that.

    Parameters
    ----------
    val : float or array-like
        The values to be conformed to the desired shape.
    shape : tuple
        The shape of the desired output at each node.
    num_input_nodes : int
        The number of nodes along which the value is repeated.

    Returns
    -------
    np.array
        The given value of the correct shape.
    """
    if np.isscalar(val) or np.prod(np.asarray(val).shape) == 1:
        shaped_val = float(val) * np.ones((num_input_nodes,) + shape)
    elif np.asarray(val).shape == shape:
        shaped_val = np.repeat(val[np.newaxis, ...], num_input_nodes, axis=0)
    else:
        shaped_val = np.reshape(val, newshape=(num_input_nodes,) + shape)
    return shaped_val


class CoerceDesvar(object):
    """
    Check the desvar options for the appropriate shape and resize accordingly with options.

    Parameters
    ----------
    num_input_nodes : int
        Number of input nodes.
    desvar_indices : ndarray
        Flattened indices of the variable.
    options : dict
        Variable options dictionary, should contain "shape".
    """
    def __init__(self, num_input_nodes, desvar_indices, options):
        self.num_input_nodes = num_input_nodes
        self.desvar_indices = desvar_indices
        self.options = options

    def __call__(self, option):
        """
        Test that an option's shape is compliant with the number of input nodes for the design variable.

        Parameters
        ----------
        option : str
            The name of the option whose value(s) are desired.

        Returns
        -------
        object
            The value of the desvar option.

        Raises
        ------
        ValueError
            If the number of values in the option is not compliant with the number of input
            nodes for the design variable.

        """
        val = self.options[option]
        if val is None or np.isscalar(val):
            return val
        # Handle value for vector/matrix valued variables
        if isinstance(val, list):
            val = np.asarray(val)
        if val.shape == self.options['shape']:
            return np.tile(val.flatten(), int(len(self.desvar_indices)/val.size))
        else:
            raise ValueError('array-valued option {0} must have same shape '
                             'as states ({1})'.format(option, self.options['shape']))


def CompWrapperConfig(comp_class):
    """
    Returns a wrapped comp_class that calls its configure_io method at the end of setup.

    This allows for standalone testing of Dymos components that normally require their parent group
    to configure them.

    Parameters
    ----------
    comp_class : Component class
       Class that we would like to wrap.

    Returns
    -------
    WrappedClass
        Wrapped version of comp_class.
    """
    class WrappedClass(comp_class):

        def setup(self):
            """
            Appends a call to configure_io after setup.
            """
            super(WrappedClass, self).setup()
            self.configure_io()

    return WrappedClass


# Modify class so we can run it standalone.
def GroupWrapperConfig(comp_class):
    """
    Returns a wrapped group_class that calls its configure_io method at the end of setup.

    This allows for standalone testing of Dymos components that normally require their parent group
    to configure them.

    Parameters
    ----------
    comp_class : Group class
       Class that we would like to wrap.

    Returns
    -------
    WrappedClass
        Wrapped version of comp_class.
    """
    class WrappedClass(comp_class):

        def setup(self):
            """
            Setup as normal.
            """
            super(WrappedClass, self).setup()

        def configure(self):
            """
            Call configure_io during configure.
            """
            self.configure_io()

    return WrappedClass
