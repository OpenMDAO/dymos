"""
Definition of ODEGroup.
"""
import re

import numpy as np
import openmdao.api as om


VAR_NAMES_REGEX = re.compile(r'([_a-zA-Z]\w*[ ]*\(?:?[.]?)')
VAR_NAMES_REGEX = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*(?:[.:][a-zA-Z_][a-zA-Z0-9_]*)+')


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
        unique_int = 0
        for expr, expr_kwargs in ode_exprs.items():
            # Pull out the var names that contain a dotted path
            # These represent ODE-relative names.
            ode_rel_paths = [x.strip() for x in re.findall(VAR_NAMES_REGEX, expr)
                                if not x.endswith('(') and not x.endswith(':')]
            # Replace each ODE-relative name with the last part of its name.
            # However, this could result in name collisions if the ODE contains
            # common variable names at different depths (e.g. aircraft.wing.mass
            # vs aircraft.payload.mass), so add a unique digit to each one.
            for rel_path in ode_rel_paths:
                unique_name = rel_path.split('.')[-1] + str(unique_int)
                expr = expr.replace(rel_path, unique_name)
                unique_int += 1
                ode_group.connect(rel_path, unique_name)
            # We need to set input defaults for things.
            # Unless otherwise specified, meta['shape'] is assumed to be (num_nodes,)
            # and meta['val'] is ones((num_nodes,)).
            # TODO: Once the delayed shape specification goes in, we can probably
            # switch to shape_by_conn.
            for var, meta in expr_kwargs.items():
                if 'shape' not in meta:
                    meta['shape'] = (nn,)
                if 'val' not in meta:
                    meta['val'] = np.ones(meta['shape'])
                ode_group.set_input_defaults(var, src_shape=meta['shape'], val=meta['val'])
            ec.add_expr(expr, **expr_kwargs)
        return ode_group


if __name__ == '__main__':
    from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

    nn = 10

    exprs = {'foo = x * y + v': {'x': {'units': 'm'}, 'y': {'units': 'm'}, 'v': {'units': 'm/s'}}}

    p = om.Problem()
    ode = make_ode(BrachistochroneODE, num_nodes=nn, ode_exprs=exprs)
    ivc = p.model.add_subsystem('ivc', om.IndepVarComp())
    ivc.add_output('x', shape=(nn,))
    ivc.add_output('y', shape=(nn,))
    p.model.add_subsystem('ode', ode)

    p.model.connect('ivc.x', 'ode.x')
    p.model.connect('ivc.y', 'ode.y')

    p.setup()
    p.final_setup()

    p.model.list_vars()

    om.n2(p)

