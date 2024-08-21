
import numpy as np

import openmdao
import openmdao.api as om
from openmdao.core.constants import _ReprClass

from .constants import INF_BOUND
from .indexing import get_desvar_indices


# unique object to check if default is given (when None is an allowed value)
_unspecified = _ReprClass("unspecified")
_none_or_unspecified = {None, _unspecified}


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

    if units is not None and time_units is not None:
        return f'{units}/{tu}'
    elif units is not None:
        return units
    elif time_units is not None:
        return f'1.0/{tu}'


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
        shaped_val = np.reshape(val, (num_input_nodes,) + shape)
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
    Return version of OpenMDAO.

    This tuple can be used for comparisons as follows:

    ```
    if om_version() < (3, 0, 0):
        # behavior if OpenMDAO is older than 3.0.0
    else:
        # behavior for newer versions
    ```

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

