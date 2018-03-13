"""Define the EqualityConstraintComp class."""

from __future__ import print_function, division, absolute_import

from numbers import Number
from six import iteritems

import numpy as np

from openmdao.api import ExplicitComponent


class EqualityConstraintComp(ExplicitComponent):
    """
    A simple equation balance for solving implicit equations.
    """

    def __init__(self, name=None, eq_units=None, lhs_name=None,
                 rhs_name=None, rhs_val=0.0, mult_name=None, mult_val=1.0,
                 out_name=None, add_constraint=True, **kwargs):
        r"""
        Initialize an EqualityConstraintComp.

        The EqualityConstraintComp is a bit like IndepVarComp in that it allows for the
        creation of one or more explicit state variables, and computes the outputs
        for those variables based on the following equation.

        .. math::

          f_{lhs}(x,...) - f_{rhs}(x,...) = f_{out}(x,...)

        Where :math:`f_{lhs}` represents the left-hand-side of the equation,
        :math:`f_{rhs}` represents the right-hand-side, and :math:`f_{mult}`
        is an optional multiplier on the left hand side.  At least one of these
        quantities should be a function of the associated state variable.  If left
        unconnected the multiplier is simply 1.0.

        New state variables, and their associated residuals are created by
        calling `add_balance`.

        A constraint for the output to be equal to 0. will be added unless `add_constraint=False`
        is provided.


        Parameters
        ----------
        name : str
            The name of the state variable to be created.
        eq_units : str or None
            Units for the left-hand-side and right-hand-side of the equation to be balanced.
        lhs_name : str or None
            Optional name for the LHS variable associated with the state variable.  If
            None, the default will be used:  'lhs:{name}'.
        rhs_name : str or None
            Optional name for the RHS variable associated with the state variable.  If
            None, the default will be used:  'rhs:{name}'.
        rhs_val : int, float, or np.array
            Default value for the RHS of the given state.  Must be compatible
            with the shape (optionally) given by the val option in kwargs.
        out_name : str or None
            Optional name for the output variable associated with the state variable. If None,
            the default will be used: `out:{name}`.
        add_constraint : bool
            If True (default), a constraint is added for the output to be zero.
        kwargs : dict
            Additional arguments to be passed for the creation of the state variable.
        """
        super(EqualityConstraintComp, self).__init__()
        self._state_vars = {}
        if name is not None:
            self.add_balance(name, eq_units, lhs_name, rhs_name, rhs_val,
                             out_name, add_constraint, **kwargs)

    def setup(self):
        """
        Define the independent variables, output variables, and partials.
        """

        for name, options in iteritems(self._state_vars):

            for s in ('lhs', 'rhs', 'out'):
                if options['{0}_name'.format(s)] is None:
                    options['{0}_name'.format(s)] = '{0}:{1}'.format(s, name)

            val = options['kwargs'].get('val', np.ones(1))
            if isinstance(val, Number):
                n = 1
            else:
                n = len(val)
            self._state_vars[name]['size'] = n

            self.add_input(options['lhs_name'],
                           val=np.ones(n),
                           units=options['eq_units'])

            self.add_input(options['rhs_name'],
                           val=options['rhs_val'] * np.ones(n),
                           units=options['eq_units'])

            self.add_output(options['out_name'],
                            val=np.zeros(n),
                            units=options['eq_units'])

            if options['add_constraint']:
                self.add_constraint(options['out_name'], equals=0.)

            ar = np.arange(n)
            self.declare_partials(of=options['out_name'], wrt=options['lhs_name'],
                                  rows=ar, cols=ar, val=1.0)
            self.declare_partials(of=options['out_name'], wrt=options['rhs_name'],
                                  rows=ar, cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        """
        Compute the output for the equality constraint.
        """
        for name, options in iteritems(self._state_vars):
            outputs[options['out_name']] = inputs[options['lhs_name']] - inputs[options['rhs_name']]

    def add_balance(self, name, eq_units=None, lhs_name=None,
                    rhs_name=None, rhs_val=0.0, out_name=None, add_constraint=True, **kwargs):
        """
        Add a new state variable and associated equation to be balanced.

        This will create new inputs `lhs:name`, `rhs:name`, and `mult:name` that will
        define the left and right sides of the equation to be balanced, and a
        multiplier for the left-hand-side.

        Parameters
        ----------
        name : str
            The name of the state variable to be created.
        eq_units : str or None
            Units for the left-hand-side and right-hand-side of the equation to be balanced.
        lhs_name : str or None
            Optional name for the LHS variable associated with the state variable.  If
            None, the default will be used:  'lhs:{name}'.
        rhs_name : str or None
            Optional name for the RHS variable associated with the state variable.  If
            None, the default will be used:  'rhs:{name}'.
        rhs_val : int, float, or np.array
            Default value for the RHS.  Must be compatible with the shape (optionally)
            given by the val option in kwargs.
        add_constraint : bool
            If True (default), a constraint is added for the output to be zero.
        out_name : str or None
            Optional name for the output variable associated with the state variable. If None,
            the default will be used: `out:{name}`.
        add_constraint : bool
            If True (default), a constraint is added for the output to be zero.
        kwargs : dict
            Additional arguments to be passed for the creation of the state variable.
        """
        self._state_vars[name] = {'kwargs': kwargs,
                                  'eq_units': eq_units,
                                  'lhs_name': lhs_name,
                                  'rhs_name': rhs_name,
                                  'rhs_val': rhs_val,
                                  'out_name': out_name,
                                  'add_constraint': add_constraint}
