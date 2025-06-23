from collections.abc import Iterable

import numpy as np

import openmdao
import openmdao.api as om
from openmdao.core.constants import _ReprClass

from .constants import INF_BOUND
from .indexing import get_desvar_indices


# unique object to check if default is given (when None is an allowed value)
_unspecified = _ReprClass("unspecified")


def is_unspecified(obj):
    """
    Return True if the object is _unspecified.

    This function should be used instead of `{obj} is _unspecified`, which
    is not reliable across processes. The use of `{obj} == _unspecified` will
    fail if `obj` is an array.

    Parameters
    ----------
    obj : any
        Any python object.

    Returns
    -------
    bool
        True if the obj is not an array, and obj == _UNDEFINED.
    """
    if isinstance(obj, Iterable):
        return False
    return obj == _unspecified


def is_none_or_unspecified(obj):
    """
    Similar to is_undefined, but also returns True if obj is None.

    Parameters
    ----------
    obj : any
        Any python object.

    Returns
    -------
    bool
        True if the obj is not an array, and obj == _UNDEFINED or obj is None.
    """
    return is_unspecified(obj) or obj is None


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

    tu = time_units if deriv == 1 else f'{time_units}**2'

    if units not in (None, 'unitless') and time_units not in (None, 'unitless'):
        return f'{units}/{tu}'
    elif units not in (None, 'unitless'):
        return units
    elif time_units not in (None, 'unitless'):
        return f'1/{tu}'
    else:  # Explicitly return None if both units and time_units are None or 'unitless'
        return None


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
        shaped_val = float(np.asarray(val).ravel()[0]) * np.ones((num_input_nodes,) + shape)
    elif np.asarray(val).shape == shape:
        shaped_val = np.repeat(val[np.newaxis, ...], num_input_nodes, axis=0)
    else:
        shaped_val = np.reshape(val, (num_input_nodes,) + shape)
    return shaped_val


def _format_phase_constraint_alias(phase, con_name, con_type, indices=None):
    """
    Get an alias for a constraint of the given type on the given path in the given phase.

    Parameters
    ----------
    phase : Phase
        The dymos phase to which the constraint belongs.
    con_name : str
        The name or path of the constraint.
    con_type : str
        One of 'initial', 'final', or 'path'.
    indices : tuple or None
        The indices of the constraint variable to be constrained. These indices
        ignore the num_nodes dimension in path constraints.

    Returns
    -------
    str
        The alias of the constraint.
    """
    str_idxs = '' if indices is None else f'[{indices}]'
    if f'.phases.{phase.name}' in phase.pathname:
        phase_path = phase.pathname.replace(f'.phases.{phase.name}', f'.{phase.name}')
    else:
        phase_path = phase.pathname
    # return f'{phase_path}->{con_type}->{con_name}{str_idxs}'
    return f'{phase_path}.{con_name}[{con_type}]{str_idxs}'


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

        if desvar_indices is None:
            desvar_indices = get_desvar_indices(size, num_input_nodes,
                                                options['fix_initial'], options['fix_final'])
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


def CompWrapperConfig(comp_class, config_io_args=None):
    """
    Returns a wrapped comp_class that calls its configure_io method at the end of setup.

    This allows for standalone testing of Dymos components that normally require their parent group
    to configure them.

    Parameters
    ----------
    comp_class : Component class
        Class that we would like to wrap.
    config_io_args : list
        Arguments to be passed to config_io.

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
            args = [] if config_io_args is None else config_io_args
            self.configure_io(*args)

    return WrappedClass


# Modify class so we can run it standalone.
def GroupWrapperConfig(comp_class, config_io_args=None):
    """
    Returns a wrapped group_class that calls its configure_io method at the end of setup.

    This allows for standalone testing of Dymos components that normally require their parent group
    to configure them.

    Parameters
    ----------
    comp_class : Group class
       Class that we would like to wrap.
    config_io_args : list
        Arguments to be passed to config_io.

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
            args = [] if config_io_args is None else config_io_args
            self.configure_io(*args)

    return WrappedClass


def create_subprob(base_name, comm, reports=False):
    """
    Create a new problem using basename possibly appended with unique identifiers if name collisions occur.

    Parameters
    ----------
    base_name : str
        The base name of the problem. This may be appended by `_{int}` to obtain a unique problem name.
        In the event of running under MPI, an 8-character hash may further append the name to ensure
        it is unique.
    comm : comm
        The MPI comm to be used by the subproblem.
    reports : bool or None or str or Sequence
        Reports setting for the subproblems run under simualate.

    Returns
    -------
    Problem
        The instantiated OpenMDAO problem instance.
    """
    from openmdao.core.problem import _problem_names

    # Find a unique sim problem name. This mostly causes problems
    # when many simulations are being run in a single process, as in testing.
    i = 0
    sim_prob_name = f'{base_name}_{i}'
    while sim_prob_name in _problem_names:
        i += 1
        sim_prob_name = f'{base_name}_{i}'

    try:
        p = om.Problem(comm=comm, reports=reports, name=sim_prob_name)
    except ValueError:
        # Testing under MPI, we still might have name collisions. In that case, add a random hash
        # to the end of the problem name.
        import hashlib
        str_hash = hashlib.sha256(used_for_security=False)[:8]
        p = om.Problem(comm=comm, reports=reports, name=f'{sim_prob_name}_{str_hash}')
    return p


def om_version():
    """
    Return version infromation for OpenMDAO.

    This information is useful for executing code that requires a specific
    version of OpenMDAO. The tuple format returned by this function can
    easily be compared using a statement like `if om_version()[0] < (3, 3, 0)`.

    Returns
    -------
    tuple
        The semantic version of OpenMDAO in a comparable tuple.
    str
        One of "dev" or "release".
    """
    try:
        numeric, rel = openmdao.__version__.split('-')
    except ValueError:
        numeric = openmdao.__version__
        rel = 'release'
    return tuple([int(s) for s in numeric.split('.')]), rel


def is_scalar_or_singleton(x):
    """
    Returns True if x is a scalar, is an instance of np.generic, or is an array of length 1.

    Parameters
    ----------
    x : float or array_like
        The variable being tested for numpy scalar-like behavior.

    Returns
    -------
    bool
        True if x is a scalr or numpy generic, or a shape=(1,) array.
    """
    return (np.isscalar(x) or
            isinstance(x, np.generic) or
            hasattr(x, 'shape') and x.shape == (1,))


def broadcast_to_nodes(a, shape, num_nodes, force_array=False):
    """
    Broadcast a to an array that is of shape (num_nodes,) + shape.

    Parameters
    ----------
    a : array_like
        The array to be broadcast.  If given as an array-like object,
        it should have the specified shape.
    shape : tuple
        The shape of the variable at each node.
    num_nodes : int
        The number of nodes to which the values of a should be broadcast.
    force_array : bool
        Return a in the shape (num_nodes,) + shape even if it is scalar-like.

    Returns
    -------
    array_like or float
        The value of a broadcast to a (num_nodes,) + shape array if it is not a scalar,
        otherwise the scalr value of a).

    Raises
    ------
    ValueError
        If the shape of the given a is neither a scalar or the shape of
    """
    if (not is_scalar_or_singleton(a) and a is not None) or (force_array and a is not None):
        _a = np.asarray(a)
        if _a.shape == shape:
            result = np.broadcast_to(_a, (num_nodes,) + shape)
        else:
            raise ValueError('array-valued scaler/ref must length equal to state-size')
    else:
        result = a
    return result


def determine_ref_ref0(ref0, ref, adder, scaler):
    r"""
    Determine proper values of ref0 and ref based on user arguments.

    Parameters
    ----------
    ref0 : float or ndarray, optional
        Value of response variable that scales to 0.0 in the driver.
    ref : float or ndarray, optional
        Value of response variable that scales to 1.0 in the driver.
    adder : float or ndarray, optional
        Value to add to the model value to get the scaled value. Adder
        is first in precedence.
    scaler : float or ndarray, optional
        Value to multiply the model value to get the scaled value. Scaler
        is second in precedence.

    Returns
    -------
    tuple
        A tuple containing ref0 and ref, properly formatted and based on scaler/adder/ref/ref0 if provided.

    Raises
    ------
    ValueError
        If both ref/ref0 and adder/scaler were provided.

    Notes
    -----
    The response can be scaled using ref and ref0.
    The argument :code:`ref0` represents the physical value when the scaled value is 0.
    The argument :code:`ref` represents the physical value when the scaled value is 1.
    """
    from openmdao.utils.general_utils import format_as_float_or_array
    # Affine scaling cannot be used with scalers/adders
    if scaler is not None or adder is not None:
        _adder = 0.0 if adder is None else adder
        _scaler = 1.0 if scaler is None else scaler

        if ref0 is not None or ref is not None:
            raise ValueError('ref/ref0 are mutually exclusive '
                             'with scaler/adder')

        # Convert scaler/adder to ref/ref0 to so we can scale an output
        ref0 = -_adder
        ref = 1.0 / _scaler - _adder

    else:
        if ref is None:
            ref = 1.0
        if ref0 is None:
            ref0 = 0.0

    ref0 = format_as_float_or_array('ref0', ref0, val_if_none=0.0, flatten=True)
    ref = format_as_float_or_array('ref', ref, val_if_none=1.0, flatten=True)

    return ref0, ref
