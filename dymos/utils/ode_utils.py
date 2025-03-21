"""
Utilities for dealing with ODE systems.
"""
from copy import deepcopy
import re

import numpy as np

import openmdao.api as om

from dymos.utils.misc import _unspecified


# This regex finds variables and any indices that follow them.
# 1 - leading underscore or letter (one)
# 2 - alphanumeric, underscore, period, or colon (zero or more)
# 3 - end of word is not followed by opening parentheses
# 4 - optional index specification in brackets
#                       1111111112222222333333334444444444444444
var_rgx = re.compile(r'([_a-zA-Z][\w.:]*\b(?!\()(\[[\d:.,-]+\])*)')


def _parse_index_string(index_str):
    """
    Convert a string representation of array indices to a tuple of slice objects,
    integer indices, or ellipsis that can be used for array access.

    Args:
        index_str (str): String representation of indices, e.g. '[1, 3]', '[1:5, 2]', '[..., 0]'

    Returns:
        tuple: A tuple containing slice objects, integers, or Ellipsis for indexing
    """
    if index_str is None:
        return None
    # Remove the outer brackets and whitespace
    clean_str = index_str.strip().strip('[]')

    if not clean_str:
        return tuple()

    # Split by comma to get each dimension's index
    dimensions = clean_str.split(',')

    result = []
    for dim in dimensions:
        dim = dim.strip()

        # Handle ellipsis
        if dim == '...' or dim == 'Ellipsis':
            result.append(Ellipsis)
            continue

        # Handle full slice ':'
        if dim == ':':
            result.append(slice(None))
            continue

        # Check if it's a slice (contains ':')
        if ':' in dim:
            # Handle slice notation (start:stop:step)
            slice_parts = dim.split(':')

            # Convert each part, handling empty strings
            start = int(slice_parts[0]) if slice_parts[0].strip() else None

            if len(slice_parts) > 1:
                stop = int(slice_parts[1]) if slice_parts[1].strip() else None
            else:
                stop = None

            if len(slice_parts) > 2:
                step = int(slice_parts[2]) if slice_parts[2].strip() else None
            else:
                step = None

            result.append(slice(start, stop, step))
        else:
            # It's a simple index
            try:
                result.append(int(dim))
            except ValueError:
                # If it's not an integer, it might be a variable
                result.append(dim)

    return tuple(result)


class ExprParser():
    """
    A special parser for Expressions in dymos.

    When invoked with a full_path as given by _extract_openmdao_variables,
    return a legal name for the variable in the expression and an indexer
    if indices were specified.
    """
    def __init__(self):
        self._unique_int = 0

    def parse(self, full_path, idx_str):
        """
        Extract the legal name and any indexer given by the full path representation.

        Parameters
        ----------
        full_path : str
            The full path of the expression variable. This may include
            dots, colons, and an index specification on the end.
        idx_str : str
            The string which follows the path specification in the expression.
            This begins and ends with brackets [].

        Returns
        -------
        tuple(str, tuple or None)
            The string containing the OpenMDAO-legal name for the variable
            in the ExecComp expression, as well as any indices specified.
        """
        is_ode_rel = False
        if '.' in full_path:
            # the path is ODE relative
            last_path = full_path.split('.')[-1]
            is_ode_rel = True
        else:
            # its a top-level variable or phase variable
            last_path = full_path
        legal_name = last_path.split(':')[-1]
        if is_ode_rel:
            legal_name += str(self._unique_int)
            self._unique_int += 1
        return legal_name, _parse_index_string(idx_str)


class ODEGroup(om.Group):
    """
    Initialize an ODEGroup to wrap the user's ODE and an ExecComp.

    Parameters
    ----------
    ode_class : class
        The class of ODE used in this Group.
    num_nodes : int
        The number of nodes used in the ODE.
    ode_init_kwargs : dict or None
        Initialization arguments for the ODE system.
    calc_exprs : dict
        The _calc_exprs dictionary of the owning phase instance.
    parameter_options : dict
        The parameter_options dictionary of the owning phase instance.
    """
    def __init__(self, ode_class, num_nodes, ode_init_kwargs=None, calc_exprs=None, parameter_options=None):
        super().__init__()
        self._ode_class = ode_class
        self._ode_init_kwargs = ode_init_kwargs or {}
        self._calc_exprs = calc_exprs or {}
        self._num_nodes = num_nodes
        self._parameter_options = parameter_options

    def setup(self):
        """
        Set up the ODEGroup.

        Adds the user's ODE class and an ExecComp.
        """
        ode_class = self._ode_class
        ode_init_kwargs = self._ode_init_kwargs
        num_nodes = self._num_nodes

        ode = ode_class(num_nodes=num_nodes, **ode_init_kwargs)

        self.add_subsystem('user_ode', ode, promotes_inputs=['*'], promotes_outputs=['*'])
        ec = om.ExecComp()
        self.add_subsystem('exec_comp', ec, promotes_inputs=['*'], promotes_outputs=['*'])

    def configure(self):
        """
        Setup up connections and promotions in the ODEGroup.
        """
        num_nodes = self._num_nodes

        seen_kwargs = set()
        parser = ExprParser()
        ec = self._get_subsystem('exec_comp')

        for expr, expr_kwargs in self._calc_exprs.items():
            common_units = _unspecified
            common_shape = _unspecified
            if 'units' in expr_kwargs:
                # units are set throughout this expression
                common_units = expr_kwargs['units']
            if 'shape' in expr_kwargs:
                common_shape = expr_kwargs['shape']

            _expr_kwargs = {}
            output_var, rhs = [s.strip() for s in expr.split('=')]
            _expr_kwargs[output_var] = deepcopy(expr_kwargs.get(output_var, {}))

            if 'shape' not in _expr_kwargs[output_var]:
                if common_shape is not _unspecified:
                    _expr_kwargs[output_var]['shape'] = (num_nodes,) + common_shape
                else:
                    _expr_kwargs[output_var]['shape'] = (num_nodes,)
            else:
                # Assume given shape is at a per-node basis
                _expr_kwargs[output_var]['shape'] = (num_nodes,) + _expr_kwargs[output_var]['shape']

            if 'units' not in _expr_kwargs[output_var]:
                if common_units is not _unspecified:
                    _expr_kwargs[output_var]['units'] = common_units

            scalar_src_idxs = []
            param_options = self._parameter_options or {}
            scalar_sources = list(param_options.keys()) + ['t_initial', 't_duration', 't_final']

            for rel_path, idx_str in re.findall(var_rgx, rhs):
                exec_var_name, src_idxs = parser.parse(rel_path, idx_str)
                expr = expr.replace(rel_path, exec_var_name)
                if '.' in rel_path:
                    self.connect(rel_path, exec_var_name, src_indices=src_idxs)

                # Only provide kwargs for things that we havent already done so.
                if exec_var_name not in seen_kwargs:
                    # Use deepcopy so we don't accidentally permanently set the shape here when we assign it.
                    _expr_kwargs[exec_var_name] = deepcopy(expr_kwargs.get(rel_path, {}))
                    if 'shape' not in _expr_kwargs[exec_var_name]:
                        if common_shape is not _unspecified:
                            _expr_kwargs[exec_var_name]['shape'] = (num_nodes,) + common_shape
                        else:
                            _expr_kwargs[exec_var_name]['shape'] = (num_nodes,)
                    else:
                        _expr_kwargs[exec_var_name]['shape'] = (num_nodes,) + _expr_kwargs[exec_var_name]['shape']

                    if exec_var_name in scalar_sources:
                        scalar_src_idxs.append(exec_var_name)

            seen_kwargs |= _expr_kwargs.keys()
            ec.add_expr(expr, **_expr_kwargs)
            if scalar_src_idxs:
                self.promotes('exec_comp', inputs=scalar_src_idxs,
                              src_indices=np.zeros(num_nodes, dtype=int))


def _make_ode_system(ode_class, num_nodes, ode_init_kwargs=None, calc_exprs=None, parameter_options=None):
    """
    Instantiate the ODE system, optionally including an ExecComp.

    Parameters
    ----------
    ode_class : System class
        The ODE class provided to the dymos phase.
    num_nodes : int
        The number of nodes at which the ODE is instantiated.
    ode_init_kwargs : dict or None
        Additional arguments we need to instantiate the ODE.
    calc_exprs : dict
        A dictionary keyed by exec comp expresions whose associated values
        are the metadata associated with the expression.
    parameter_options : dict
        The parameter options of the owning phase.

    Returns
    -------
    ode : System
        The instantiation of ode_class.  If `calc_exprs` were given, it
        returns a group that wraps the instantiated ode_class, as well
        as an exec comp used to evaluate the expressions.
    """
    _kwargs = ode_init_kwargs or {}
    if not calc_exprs:
        return ode_class(num_nodes=num_nodes, **_kwargs)
    else:
        ode_group = ODEGroup(ode_class,
                             num_nodes=num_nodes,
                             ode_init_kwargs=ode_init_kwargs,
                             calc_exprs=calc_exprs,
                             parameter_options=parameter_options)
        return ode_group
