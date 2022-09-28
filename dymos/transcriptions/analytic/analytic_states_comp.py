"""Definition of the AnalyticStatesComp Component."""


import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from ...utils.misc import _unspecified
from ...options import options as dymos_options


class AnalyticStatesComp(ExplicitComponent):
    """
    A component which simply passes an input initial value of a state to an equivalent output.

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

    def add_state(self, name, options):
        """
        Add an input/output pair for a variable to this component.

        Parameters
        ----------
        name : str
            Name of the input variable in this component's namespace.
        options : OptionsDictionary
            The StateOptionsDictionary associated with the state.

        Returns
        -------
        input_meta : dict
            The metadata associated with the input.
        output_meta : dict
            The metadata associated with the output.
        """
        # _out_name = output_name if output_name is not None else f'initial_state_vals:{name}'

        shape = options['shape']
        val = np.asarray(options['val'])
        units = options['units']

        if shape in {None, _unspecified}:
            _shape = (1,)
            size = _val.size
        else:
            _shape = shape
            size = np.prod(shape)
        _out_shape = (1,) + tuple(shape)
        ar = np.arange(size, dtype=int)

        if np.ndim(val) == 0 or val.shape == (1,):
            in_val = np.full(_shape, val)
        else:
            in_val = val
        out_val = np.expand_dims(in_val, axis=0)

        if options['input_initial']:
            i_meta = self.add_input(name=f'initial_states:{name}', val=in_val, shape=_shape, units=units)
            o_meta = self.add_output(name=f'initial_state_vals:{name}', val=out_val, shape=_out_shape, units=units)

            self.declare_partials(of=f'initial_state_vals:{name}', wrt=f'initial_states:{name}',
                                  rows=ar, cols=ar, val=1.0)

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
