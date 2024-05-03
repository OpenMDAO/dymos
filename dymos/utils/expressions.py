import re

from openmdao.components.exec_comp import VAR_RGX


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
    names = [x.strip() for x in re.findall(VAR_RGX, s) if not x.startswith('.')]
    vnames: set[str] = set()
    for n in names:
        if n.endswith('('):
            continue
        vnames.add(n)

    return vnames
