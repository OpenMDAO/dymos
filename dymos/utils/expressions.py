from copy import deepcopy
import re

import openmdao.api as om
from openmdao.core.system import System
from openmdao.components.exec_comp import VAR_RGX

# regex to check for variable names.
VAR_RGX_WITH_SEPS = re.compile(r'([.]*[_a-zA-Z][\w\.]*[ ]*\(?)')


def parse_expression(expr: str) -> tuple[set[str], set[str]]:
    """
    Parse the given expression into output names, variable names, and function names.

    Parameters
    ----------
    expr : str
        An expression string containing a variable assignment.

    Returns
    -------
    output_names : list
        The names of the outputs created by the expression evaluation.
    variable_names : list
        The names of the variables used in the expression evaluation.
    """
    lhs, _, rhs = expr.partition('=')
    onames = _parse_for_out_vars(lhs)
    vnames = _parse_for_names(rhs)
    return onames, vnames


def _parse_for_out_vars(s: str) -> set[str]:
    onames: set[str] = set([x.strip() for x in re.findall(VAR_RGX, s)
                            if not x.endswith('(') and not x.startswith('.')])
    return onames


def _parse_for_names(s: str) -> set[str]:
    names = [x.strip() for x in re.findall(VAR_RGX_WITH_SEPS, s) if not x.startswith('.')]
    vnames: set[str] = set()
    for n in names:
        if n.endswith('('):
            continue
        vnames.add(n)

    return vnames


def add_exec_comp_to_ode_group(ode_group: System,
                               exprs: list[tuple[str, dict[str, object]]],
                               num_nodes: int):
    """
    Add an ExecComp to the given ODE group to handle user-defined calculations.

    Parameters
    ----------
    ode_group : System
        The Group which contains the user ODE and the ExecComp.
    exprs : list
        A list of expression, kwarg pairs to be added to the ExecComp.
    num_nodes : _type_
        The number of nodes at which the ODE is being evaluated.
    """
    exec_comp = ode_group.add_subsystem('expr_ode',
                                        subsys=om.ExecComp(),
                                        promotes_inputs=['*'],
                                        promotes_outputs=['*'])
    used_kwargs = set()
    inputs = {}
    for expr, kwargs in deepcopy(exprs):
        onames, vnames = parse_expression(expr)
        for varname in onames.union(vnames):
            if varname in vnames:
                inputs[varname.split('.')[-1]] = varname
            if varname not in kwargs:
                # No kwargs given for variable. Default shape to num_nodes.
                kwargs[varname] = {'shape': (num_nodes,)}
            elif 'shape' not in kwargs[varname]:
                # kwargs given, but not shape. Default shape to num_nodes.
                kwargs[varname]['shape'] = (num_nodes,)
            else:
                # shape given. Prepend num_nodes
                kwargs[varname]['shape'] = (num_nodes,) + kwargs[varname]['shape']
        new_kwargs = {k: v for k, v in kwargs.items() if k not in used_kwargs}
        new_expr = expr
        for var, dotted_var in inputs.items():
            if dotted_var != var:
                new_expr = new_expr.replace(dotted_var, var)
                new_kwargs[var] = new_kwargs[dotted_var]
                new_kwargs.pop(dotted_var)
        exec_comp.add_expr(new_expr, **new_kwargs)
        used_kwargs |= kwargs.keys()
    # For every dotted-input in an expression, explicitly connect it to its source.
    for inp, source in inputs.items():
        if '.' in source:
            ode_group.connect(source, inp)
