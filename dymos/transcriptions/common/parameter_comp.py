"""Definition of the Passthru Component."""


import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from dymos.utils.misc import is_none_or_unspecified
from dymos._options import options as dymos_options


class ParameterComp(ExplicitComponent):
    """
    A component which simply passes a parameter input to an equivalent output.

    Parameters
    ----------
    time_options : TimeOptionsDictionary or None
        If None, specify time options for the creation of t_initial and t_duration inputs and outputs.
    **kwargs : dict
        Arguments to be passed to the component initialization method.

    Attributes
    ----------
    _vars : dict
        Container mapping name of variables to be muxed with additional data.
    _input_names : dict
        Container mapping name of variables to be muxed with associated inputs.
    """

    def __init__(self, time_options=None, **kwargs):
        """
        Instantiate MuxComp and populate private members.
        """
        super().__init__(**kwargs)
        self.time_options = time_options

        self._no_check_partials = not dymos_options['include_check_partials']

    def setup(self):
        """
        Add time-related I/O to the ParameterComp at setup, if provided.
        """
        time_options = self.time_options

        if time_options:
            ti_val = self.time_options['initial_val']
            td_val = self.time_options['duration_val']
            units = self.time_options['units']

            self.add_input(name='t_initial', val=ti_val, shape=(1,), units=units)
            self.add_output(name='t_initial_val', val=ti_val, shape=(1,), units=units)
            self.declare_partials(of='t_initial_val', wrt='t_initial', val=1.0)

            self.add_input(name='t_duration', val=td_val, shape=(1,), units=units)
            self.add_output(name='t_duration_val', val=td_val, shape=(1,), units=units)
            self.declare_partials(of='t_duration_val', wrt='t_duration', val=1.0)

    def add_parameter(self, name, val=1.0, shape=None, output_name=None,
                      units=None, desc='', tags=None, input_tags=None, output_tags=None, input_shape_by_conn=False,
                      input_copy_shape=None, output_shape_by_conn=False, output_copy_shape=None,
                      distributed=None, res_units=None, lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0,):
        """
        Add an input/output pair for a variable to this component.

        Parameters
        ----------
        name : str
            Name of the input variable in this component's namespace.
        val : float or list or tuple or ndarray or Iterable
            The initial value of the variable being added in user-defined units.
            Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if src_indices not provided and
            val is not an array. Default is None.
        output_name : str
            Name given to the output variale in this component's namespace.  If None, f'{name}_value' is used.
        units : str or None
            Units in which this input variable will be provided to the component
            during execution. Default is None, which means it is unitless.
        desc : str
            Description of the variable.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs.
        input_tags : str or list of strs
            User defined tags applied only to the inputs.
        output_tags : str or list of strs
            User defined tags applied only to the outputs.
        input_shape_by_conn : bool
            If True, shape this input to match its connected output.
        input_copy_shape : str or None
            If a str, that str is the name of a variable. Shape this input to match that of
            the named variable.
        output_shape_by_conn : bool
            If True, shape this output to match its connected input(s).
        output_copy_shape : str or None
            If a str, that str is the name of a variable. Shape this output to match that of
            the named variable.
        distributed : bool
            If True, this variable is a distributed variable, so it can have different sizes/values
            across MPI processes.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        lower : float or list or tuple or ndarray or Iterable or None
            Lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or or Iterable None
            Upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float or ndarray
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float or ndarray
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float or ndarray
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is 1.

        Returns
        -------
        input_meta : dict
            The metadata associated with the input.
        output_meta : dict
            The metadata associated with the output.
        """
        _out_name = output_name if output_name is not None else f'parameter_vals:{name}'

        _val = np.asarray(val)

        if is_none_or_unspecified(shape):
            _shape = (1,)
            size = _val.size
        else:
            _shape = shape
            size = np.prod(shape)
        _out_shape = (1,) + tuple(shape)
        ar = np.arange(size, dtype=int)

        if tags is None:
            tags = []
        elif isinstance(tags, str):
            tags = [tags]

        if input_tags is None:
            input_tags = []
        elif isinstance(input_tags, str):
            input_tags = [input_tags]

        if output_tags is None:
            output_tags = []
        elif isinstance(output_tags, str):
            output_tags = [output_tags]

        in_val = _val
        out_val = np.expand_dims(in_val, axis=0)

        i_meta = self.add_input(name=f'parameters:{name}', val=in_val, shape=_shape, units=units, desc=desc,
                                tags=tags + input_tags,
                                shape_by_conn=input_shape_by_conn, copy_shape=input_copy_shape, distributed=distributed)

        o_meta = self.add_output(name=_out_name, val=out_val, shape=_out_shape, units=units,
                                 desc=desc, tags=tags + output_tags, res_units=res_units, ref=ref, ref0=ref0,
                                 res_ref=res_ref, lower=lower, upper=upper, shape_by_conn=output_shape_by_conn,
                                 copy_shape=output_copy_shape, distributed=distributed)

        self.declare_partials(of=_out_name, wrt=f'parameters:{name}', rows=ar, cols=ar, val=1.0)

        return i_meta, o_meta

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Mux the inputs into the appropriate outputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : Vector
            Discrete input variables read via discrete_inputs[key].
        discrete_outputs : Vector
            Discrete output variables read via discrete_outputs[key].
        """
        outputs.set_val(inputs.asarray())
