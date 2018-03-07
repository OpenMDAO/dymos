from __future__ import division, print_function

import numpy as np
from six import iteritems

from openmdao.api import ExplicitComponent


class BoundaryConstraintComp(ExplicitComponent):

    def __init__(self, name=None, val=1.0, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        name : str or None or [(str, value), ...] or [(str, value, kwargs), ...]
            name of the variable.
            If None, variables should be defined external to this class by calling add_output.
            For backwards compatibility with OpenMDAO v1, this can also be a list of tuples
            in the case of declaring multiple variables at once.
        val : float or ndarray
            value of the variable if a single variable is being defined.
        **kwargs : dict
            keyword arguments.
        """
        super(BoundaryConstraintComp, self).__init__()
        self._initial_constraints = []
        self._final_constraints = []
        self._vars = {}

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        for (name, kwargs) in self._initial_constraints:
            input_name = 'boundary_values:{0}'.format(name)
            output_name = 'initial_value:{0}'.format(name)
            self._vars[input_name] = {'input_name': input_name,
                                      'initial_constraint': output_name,
                                      'final_constraint': '',
                                      'shape': kwargs['shape']}

            # self._vars.append((input_name, output_name, kwargs['shape']))

            input_kwargs = {k: kwargs[k] for k in ('units', 'shape', 'desc', 'var_set')}
            input_kwargs['shape'] = tuple([2] + list(input_kwargs['shape']))
            self.add_input(input_name, **input_kwargs)

            output_kwargs = {k: kwargs[k] for k in ('units', 'shape', 'desc', 'var_set')}
            self.add_output(output_name, **output_kwargs)

            constraint_kwargs = {k: kwargs.get(k, None)
                                 for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
                                           'scaler', 'indices', 'linear')}
            self.add_constraint(output_name, **constraint_kwargs)

        for (name, kwargs) in self._final_constraints:
            input_name = 'boundary_values:{0}'.format(name)
            output_name = 'final_value:{0}'.format(name)

            if input_name in self._vars:
                self._vars[input_name]['final_constraint'] = output_name
            else:
                self._vars[input_name] = {'input_name': input_name,
                                          'initial_constraint': '',
                                          'final_constraint': output_name,
                                          'shape': kwargs['shape']}
                input_kwargs = {k: kwargs[k] for k in ('units', 'shape', 'desc', 'var_set')}
                input_kwargs['shape'] = tuple([2] + list(input_kwargs['shape']))
                self.add_input(input_name, **input_kwargs)

            output_kwargs = {k: kwargs[k] for k in ('units', 'shape', 'desc', 'var_set')}
            self.add_output(output_name, **output_kwargs)

            constraint_kwargs = {k: kwargs.get(k, None)
                                 for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
                                           'scaler', 'indices', 'linear')}
            self.add_constraint(output_name, **constraint_kwargs)

        # Setup partials
        for input_name, options in iteritems(self._vars):
            size = np.prod(options['shape'])

            rs = np.arange(size)
            cs_initial = np.arange(size)
            cs_final = size + np.arange(size)

            if options['initial_constraint']:
                self.declare_partials(of=options['initial_constraint'],
                                      wrt=input_name,
                                      val=np.ones(size),
                                      rows=rs,
                                      cols=cs_initial)

            if options['final_constraint']:
                self.declare_partials(of=options['final_constraint'],
                                      wrt=input_name,
                                      val=np.ones(size),
                                      rows=rs,
                                      cols=cs_final)

    def compute(self, inputs, outputs):

        for input_name, options in iteritems(self._vars):
            if options['initial_constraint']:
                outputs[options['initial_constraint']] = inputs[input_name][0, ...]
            if options['final_constraint']:
                outputs[options['final_constraint']] = inputs[input_name][-1, ...]

    def _add_initial_constraint(self, name, shape=(1,), units=None, res_units=None, desc='',
                                lower=None, upper=None, equals=None,
                                scaler=None, adder=None, ref=1.0, ref0=0.0,
                                linear=False, res_ref=1.0, var_set=0, distributed=False):
        """
        Add an initial constraint to this component

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable
        lower : float or list or tuple or ndarray or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        scaler : float or None
            A multiplicative scaler on the constraint value for the optimizer.
        adder : float or None
            A parameter which is added to the value before scaler is applied to produce
            the value seen by the optimizer.
        ref : float or None
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float or None
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        linear : bool
            True if the *total* derivative of the constrained variable is linear, otherwise False.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        distributed : bool
            If True, this variable is distributed across multiple processes.
        """
        kwargs = {'shape': shape, 'units': units, 'res_units': res_units, 'desc': desc,
                  'lower': lower, 'upper': upper, 'equals': equals,
                  'scaler': scaler, 'adder': adder, 'ref': ref, 'ref0': ref0, 'linear': linear,
                  'res_ref': res_ref, 'var_set': var_set, 'distributed': distributed}
        self._initial_constraints.append((name, kwargs))

    def _add_final_constraint(self, name, shape=(1,), units=None, res_units=None, desc='',
                              lower=None, upper=None, equals=None,
                              scaler=None, adder=None, ref=None, ref0=None,
                              linear=False, res_ref=1.0, var_set=0, distributed=False):
        """
        Add a final constraint to this component

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable
        lower : float or list or tuple or ndarray or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        scaler : float or None
            A multiplicative scaler on the constraint value for the optimizer. Default is None.
        adder : float or None
            A parameter which is added to the value before scaler is applied to produce
            the value seen by the optimizer. Default is None.
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is None.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is None.
        linear : bool
            True if the *total* derivative of the constrained variable is linear, otherwise False.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        distributed : bool
            If True, this variable is distributed across multiple processes.
        """
        kwargs = {'shape': shape, 'units': units, 'res_units': res_units, 'desc': desc,
                  'lower': lower, 'upper': upper, 'equals': equals,
                  'scaler': scaler, 'adder': adder, 'ref': ref, 'ref0': ref0, 'linear': linear,
                  'res_ref': res_ref, 'var_set': var_set, 'distributed': distributed}
        self._final_constraints.append((name, kwargs))
