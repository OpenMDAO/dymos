import numpy as np

import openmdao.api as om

from dymos.utils.misc import get_rate_units
from dymos.options import options as dymos_options


class StateRateCollectorComp(om.ExplicitComponent):
    """
    Class definition for StateRateCollectorComp.

    Collects the state rates and outputs them in the units specified in the state options.
    For explicit integration this is necessary when the output providing the state rate has
    different units than those defined in the state_options/time_options.

    Parameters
    ----------
    vec_size : int
        The number of points in the first dimension of each input/output. This is typically called
        num_nodes in Dymos, but in the case of ExplicitIntegration this vector size is related to
        the number of stages in a step, and not the number of nodes in the transcription.
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, vec_size=1, **kwargs):
        super().__init__(**kwargs)

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_names = {}

        self._vec_size = vec_size

        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of options for the ODE state variables.')
        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')

    def configure_io(self):
        """
        Create inputs/outputs on this component.
        """
        vec_size = self._vec_size
        state_options = self.options['state_options']
        time_units = self.options['time_units']

        for name, options in state_options.items():
            self._input_names[name] = f'state_rates_in:{name}_rate'
            self._output_names[name] = f'state_rates:{name}_rate'
            shape = options['shape']
            size = np.prod(shape, dtype=int)
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_input(self._input_names[name], shape=(vec_size,) + shape, units=rate_units)
            self.add_output(self._output_names[name], shape=(vec_size,) + shape, units=rate_units)

            ar = np.arange(vec_size*size, dtype=int)
            self.declare_partials(of=self._output_names[name],
                                  wrt=self._input_names[name],
                                  rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete outputs.
        """
        state_options = self.options['state_options']

        for name, options in state_options.items():
            outputs[self._output_names[name]] = inputs[self._input_names[name]]
