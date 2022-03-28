"""Definition of the Passthru Component."""


import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from ...utils.misc import _unspecified
from ...options import options as dymos_options


class ParameterComp(ExplicitComponent):
    """
    A component which simply passes a parameter input to an equivalent output.

    Parameters
    ----------
    **kwargs : dict
        Arguments to be passed to the component initialization method.

    Attributes
    ----------
    _vars : dict
        Container mapping name of variables to be muxed with additional data.
    _input_names : dict
        Container mapping name of variables to be muxed with associated inputs.
    """

    def __init__(self, **kwargs):
        """
        Instantiate MuxComp and populate private members.
        """
        super().__init__(**kwargs)

        self._no_check_partials = not dymos_options['include_check_partials']

    def add_parameter(self, name, val=1.0, shape=None, output_name=None, src_indices=None, flat_src_indices=None,
                      units=None, desc='', tags=None, input_tags=None, output_tags=None, input_shape_by_conn=False,
                      input_copy_shape=None, output_shape_by_conn=False, output_copy_shape=None,
                      distributed=None, res_units=None, lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=1.0, ):
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
        src_indices : int or list or tuple or int ndarray or Iterable or None
            The global indices of the source variable to transfer data from.
            A value of None implies this input depends on all entries of the source array.
            Default is None. The shapes of the target and src_indices must match,
            and the form of the entries within is determined by the value of 'flat_src_indices'.
        flat_src_indices : bool
            If True and the source is non-flat, each entry of src_indices is assumed to be an index
            into the flattened source.  Ignored if the source is flat.
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

        if shape in (None, _unspecified):
            _shape = (1,)
            size = np.asarray(val).size
        else:
            _shape = shape
            size = np.prod(shape)
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

        i_meta = self.add_input(name=f'parameters:{name}', val=val, shape=_shape, units=units, desc=desc,
                                src_indices=src_indices, flat_src_indices=flat_src_indices, tags=tags + input_tags,
                                shape_by_conn=input_shape_by_conn, copy_shape=input_copy_shape, distributed=distributed)

        o_meta = self.add_output(name=_out_name, val=val, shape=_shape, units=units, desc=desc, tags=tags + output_tags,
                                 res_units=res_units, ref=ref, ref0=ref0, res_ref=res_ref, lower=lower, upper=upper,
                                 shape_by_conn=output_shape_by_conn, copy_shape=output_copy_shape,
                                 distributed=distributed)

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
