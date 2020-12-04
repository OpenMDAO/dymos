import numpy as np

import openmdao.api as om

from ...utils.constants import INF_BOUND
from ...options import options as dymos_options


class BoundaryConstraintComp(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        self.options.declare('loc', values=('initial', 'final'),
                             desc='the location in the phase of this boundary constraint '
                                  '(either \'initial\' or \'final\'')
        self._constraints = []
        self._vars = {}

    def configure_io(self):
        """
        Define the independent variables as output variables.

        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        for (name, kwargs) in self._constraints:
            input_name = '{0}_value_in:{1}'.format(self.options['loc'], name)
            output_name = '{0}_value:{1}'.format(self.options['loc'], name)
            self._vars[name] = {'input_name': input_name,
                                'output_name': output_name,
                                'shape': kwargs['shape']}

            input_kwargs = {k: kwargs[k] for k in ('units', 'shape', 'desc')}
            self.add_input(input_name, **input_kwargs)

            output_kwargs = {k: kwargs[k] for k in ('units', 'shape', 'desc')}
            self.add_output(output_name, **output_kwargs)

            constraint_kwargs = {k: kwargs.get(k, None)
                                 for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
                                           'scaler', 'indices', 'linear')}
            self.add_constraint(output_name, **constraint_kwargs)

        # Setup partials
        for name, options in self._vars.items():
            size = int(np.prod(options['shape']))

            rs = np.arange(size)
            cs = np.arange(size)

            self.declare_partials(of=options['output_name'],
                                  wrt=options['input_name'],
                                  val=np.ones(size),
                                  rows=rs,
                                  cols=cs)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        for name, options in self._vars.items():
            outputs[options['output_name']] = inputs[options['input_name']]

    def _add_constraint(self, name, units=None, res_units=None, desc='',
                        shape=None, indices=None, flat_indices=True,
                        lower=None, upper=None, equals=None,
                        scaler=None, adder=None, ref=1.0, ref0=0.0,
                        linear=False, res_ref=1.0, distributed=False):
        """
        Add a constraint to this component

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        indices : tuple, list, ndarray, or None
            The indices of the output variable to be boundary constrained.  If provided, the
            resulting constraint is always a 1D vector with the number of elements provided in
            indices.  Indices should be a 1D sequence of tuples, each providing an index into the
            source output if flat_indices is False, or integers if flat_indices is True.
        flat_indices : bool
            Whether or not indices is provided as 'flat' indices per OpenMDAO's flat_source_indices
            option when connecting variables.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
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
        distributed : bool
            If True, this variable is distributed across multiple processes.
        """
        lower = -INF_BOUND if upper is not None and lower is None else lower
        upper = INF_BOUND if lower is not None and upper is None else upper
        kwargs = {'units': units, 'res_units': res_units, 'desc': desc,
                  'shape': shape, 'indices': indices, 'flat_indices': flat_indices,
                  'lower': lower, 'upper': upper, 'equals': equals,
                  'scaler': scaler, 'adder': adder, 'ref': ref, 'ref0': ref0, 'linear': linear,
                  'res_ref': res_ref, 'distributed': distributed}
        self._constraints.append((name, kwargs))
