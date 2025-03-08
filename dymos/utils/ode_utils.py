"""
Utilities for dealing with ODE systems.
"""
from copy import deepcopy
import re

import openmdao.api as om

from openmdao.core.constants import _UNDEFINED
from openmdao.api import is_undefined


# VAR_NAMES_REGEX = re.compile(r'([_a-zA-Z]\w*[ ]*\(?:?[.]?)')
VAR_NAMES_REGEX = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*(?:[.:][a-zA-Z_][a-zA-Z0-9_]*)+')


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


def _extract_vars(equation):
    """
    Extract variables from an expression

    Unlike the standard ExecComp, we allow the use of ode-relative, dotted paths
    in expressions. This function handles several special circumstances:
    - variables that include dots or colons
    - variables that include array indices (e.g. "x[1, 3]")
    - variables within the arguments of arbitrary functions

    Parameters
    ----------
    equation : str
        The equation (or right hand side) from which we're extracting variables.

    Returns
    -------
    str
        The variables extracted. These may include indices, dots or colons.
    """
    variables = []
    
    # Get the assignment variable (left side)
    parts = equation.split('=', 1)
    if len(parts) > 1:
        left_var = parts[0].strip()
        if re.match(r'[a-zA-Z_][a-zA-Z0-9_]*$', left_var):
            variables.append(left_var)
    
    # Get variables with array indices - full path.name:var[idx] pattern
    indexed_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*(?:[.:][a-zA-Z_][a-zA-Z0-9_]*)*\[[^]]*\])'
    indexed_vars = re.findall(indexed_pattern, equation)
    variables.extend(indexed_vars)
    
    # Get function arguments
    func_args_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*\s*\(([^)]+)\)'
    for match in re.finditer(func_args_pattern, equation):
        args = match.group(1).strip()
        # Extract variable names from function arguments
        arg_vars = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', args)
        variables.extend(arg_vars)
    
    # Get simple variables (excluding function names)
    # This pattern looks for standalone variables not followed by ( or part of a path
    simple_pattern = r'(?<![.:a-zA-Z0-9_\[])[a-zA-Z_][a-zA-Z0-9_]*(?![.:a-zA-Z0-9_\[\(])'
    simple_vars = re.findall(simple_pattern, equation)
    variables.extend(simple_vars)
    
    # Remove duplicates while preserving order
    result = []
    seen = set()
    for var in variables:
        if var not in seen:
            seen.add(var)
            result.append(var)
    
    return result


def make_ode(ode_class, num_nodes, ode_init_kwargs=None, ode_exprs=None):
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
    ode_exprs : dict
        A dictionary keyed by exec comp expresions whose associated values
        are the metadata associated with the expression.

    Returns
    -------
    ode : System
        The instantiation of ode_class.  If `ode_exprs` were given, it
        returns a group that wraps the instantiated ode_class, as well
        as an exec comp used to evaluate the expressions.
    """
    _kwargs = ode_init_kwargs or {}
    ode = ode_class(num_nodes=num_nodes, **_kwargs)
    if not ode_exprs:
        return ode
    else:
        ode_group = om.Group()
        ode_group.add_subsystem('user_ode', ode, promotes=['*'])
        ec = om.ExecComp()
        ode_group.add_subsystem('exec_comp', ec, promotes=['*'])
        parser = ExprParser()

        seen_kwargs = set()

        # This regex finds variables and any indices that follow them.
        # 1 - leading underscore or letter (one)
        # 2 - alphanumeric, underscore, period, or colon (zero or more)
        # 3 - end of word is not followed by opening parentheses
        # 4 - optional index specification in brackets
        var_rgx = re.compile(r'([_a-zA-Z][\w.:]*\b(?!\()(\[[\d:.,-]+\])*)')
        #                       11111111122222223333333344444444444444444

        for expr, expr_kwargs in ode_exprs.items():
            common_units = _UNDEFINED
            common_shape = _UNDEFINED
            if 'units' in expr_kwargs:
                # units are set throughout this expression
                common_units = expr_kwargs['units']
            if 'shape' in expr_kwargs:
                common_shape = expr_kwargs['shape']

            _expr_kwargs = {}
            output_var, rhs = [s.strip() for s in expr.split('=')]
            _expr_kwargs[output_var] = deepcopy(expr_kwargs.get(output_var, {}))

            if 'shape' not in _expr_kwargs[output_var]:
                if not is_undefined(common_shape):
                    _expr_kwargs[output_var]['shape'] = (num_nodes,) + common_shape
                else:
                    _expr_kwargs[output_var]['shape'] = (num_nodes,)
            else:
                # Assume given shape is at a per-node basis
                _expr_kwargs[output_var]['shape'] = (num_nodes,) + _expr_kwargs[output_var]['shape']
            
            if 'units' not in _expr_kwargs[output_var]:
                if not is_undefined(common_units):
                    _expr_kwargs[output_var]['units'] = common_units

            for rel_path, idx_str in re.findall(var_rgx, rhs):
                exec_var_name, src_idxs = parser.parse(rel_path, idx_str)
                expr = expr.replace(rel_path, exec_var_name)
                if '.' in rel_path:
                    ode_group.connect(rel_path, exec_var_name, src_indices=src_idxs)
                
                # Only provide kwargs for things that we havent already done so.
                if exec_var_name not in seen_kwargs:
                    _expr_kwargs[exec_var_name] = expr_kwargs.get(rel_path, {})
                    if 'shape' not in _expr_kwargs[exec_var_name]:
                        if not is_undefined(common_shape):
                            _expr_kwargs[exec_var_name]['shape'] = (num_nodes,) + common_shape
                        else:
                            _expr_kwargs[exec_var_name]['shape'] = (num_nodes,)
                    else:
                        _expr_kwargs[exec_var_name]['shape'] = (num_nodes,) + _expr_kwargs[exec_var_name]['shape']

            seen_kwargs |= _expr_kwargs.keys()
            ec.add_expr(expr, **_expr_kwargs)
        return ode_group


if __name__ == '__main__':
    from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

    nn = 10

    exprs = {'foo = xdot + vdot': {'foo': {'units': 'm/s', 'copy_shape': 'xdot'}}}

    p = om.Problem()
    ode = make_ode(BrachistochroneODE, num_nodes=nn, ode_exprs=exprs)
    ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
    ivc.add_output('x', shape=(nn,), units='m')
    ivc.add_output('y', shape=(nn,), units='m')
    p.model.add_subsystem('ode', ode)

    p.setup()
    p.final_setup()
    p.run_model()

    p.model.list_vars(units=True)
