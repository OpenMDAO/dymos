from collections.abc import Iterable

import numpy as np

from .constants import INF_BOUND


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
    def __init__(self, num_input_nodes, desvar_indices=None, options=None):
        self.num_input_nodes = num_input_nodes
        shape = options['shape']
        size = np.prod(shape, dtype=int)
        fix_initial = options['fix_initial']
        fix_final = options['fix_final']

        if desvar_indices is None:
            desvar_indices = list(range(size * num_input_nodes))

            if fix_initial:
                if isinstance(fix_initial, Iterable):
                    idxs_to_fix = np.where(np.asarray(fix_initial))[0]
                    for idx_to_fix in reversed(sorted(idxs_to_fix)):
                        del desvar_indices[idx_to_fix]
                else:
                    del desvar_indices[:size]

            if fix_final:
                if isinstance(fix_final, Iterable):
                    idxs_to_fix = np.where(np.asarray(fix_final))[0]
                    for idx_to_fix in reversed(sorted(idxs_to_fix)):
                        del desvar_indices[-size + idx_to_fix]
                else:
                    del desvar_indices[-size:]

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
        if option == 'lower':
            lb = np.zeros_like(self.desvar_indices, dtype=float)
            lb[:] = -INF_BOUND if self.options['lower'] is None else val
            return lb

        if option == 'upper':
            ub = np.zeros_like(self.desvar_indices, dtype=float)
            ub[:] = INF_BOUND if self.options['upper'] is None else val
            return ub

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
