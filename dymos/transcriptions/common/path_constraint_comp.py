from __future__ import division, print_function

import numpy as np
import openmdao.api as om

from dymos.utils.constants import INF_BOUND


class PathConstraintComp(om.ExplicitComponent):

    def initialize(self):
        self._path_constraints = []
        self._vars = []
        self.options.declare('num_nodes', types=(int,), desc='The number of nodes in the phase '
                                                             'at which the path constraint is to '
                                                             'be evaluated')

    def _add_path_constraint(self, name, var_class, shape=None, units=None, res_units=None, desc='',
                             indices=None, lower=None, upper=None, equals=None, scaler=None,
                             adder=None, ref=None, ref0=None, linear=False, res_ref=1.0, type_=None,
                             distributed=False):
        """
        Add a final constraint to this component

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        var_class : str
            The 'class' of the variable as given by phase.classify_var.  One of 'time', 'state',
            'indep_control', 'input_control', 'design_parameter', 'input_parameter',
            'control_rate', 'control_rate2', or 'ode'.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        indices : tuple or list or ndarray or None
            Indices that specify which elements in shape are to be path constrained.  If None,
            then the constraint will apply to all values and lower/upper/equals must be scalar
            or of the same shape.  Indices assumes C-order flattening.  For instance, if
            constraining element [0, 1] of a variable with shape [2, 2], indices=[3].
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
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        linear : bool
            True if the *total* derivative of the constrained variable is linear, otherwise False.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.
        type_ : str
            The kind of variable be constrained, as returned by classify_var.
        distributed : bool
            If True, this variable is distributed across multiple processes.
        """
        src_all = var_class in ['time', 'time_phase', 'indep_control', 'input_control',
                                'control_rate', 'control_rate2', 'indep_polynomial_control',
                                'input_polynomial_control', 'polynomial_control_rate',
                                'polynomial_control_rate2', 'design_parameter', 'input_parameter']

        lower = -INF_BOUND if upper is not None and lower is None else lower
        upper = INF_BOUND if lower is not None and upper is None else upper
        kwargs = {'shape': shape, 'units': units, 'res_units': res_units, 'desc': desc,
                  'indices': indices, 'lower': lower, 'upper': upper, 'equals': equals,
                  'scaler': scaler, 'adder': adder, 'ref': ref, 'ref0': ref0, 'linear': linear,
                  'src_all': src_all, 'res_ref': res_ref, 'distributed': distributed,
                  'type_': type_}
        self._path_constraints.append((name, kwargs))

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        num_nodes = self.options['num_nodes']

        for (name, kwargs) in self._path_constraints:
            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            input_name = 'all_values:{0}'.format(name)
            self.add_input(input_name,
                           shape=(num_nodes,) + kwargs['shape'],
                           **input_kwargs)

            output_name = 'path:{0}'.format(name)
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (num_nodes,) + kwargs['shape']
            self.add_output(output_name, **output_kwargs)

            constraint_kwargs = {k: kwargs.get(k, None)
                                 for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
                                           'scaler', 'indices', 'linear')}

            # Convert indices from those in one time instance to those in all time instances
            template = np.zeros(np.prod(kwargs['shape']), dtype=int)
            template[kwargs['indices']] = 1
            template = np.tile(template, num_nodes)
            constraint_kwargs['indices'] = np.nonzero(template)[0]

            self.add_constraint(output_name, **constraint_kwargs)

            self._vars.append((input_name, output_name, kwargs['shape']))

            # Setup partials
            all_shape = (num_nodes,) + kwargs['shape']
            var_size = np.prod(kwargs['shape'])
            all_size = np.prod(all_shape)

            all_row_starts = np.arange(num_nodes, dtype=int) * var_size
            all_rows = []
            for i in all_row_starts:
                all_rows.extend(range(i, i + var_size))
            all_rows = np.asarray(all_rows, dtype=int)

            self.declare_partials(
                of=output_name,
                wrt=input_name,
                dependent=True,
                rows=all_rows,
                cols=np.arange(all_size),
                val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = inputs[input_name]
