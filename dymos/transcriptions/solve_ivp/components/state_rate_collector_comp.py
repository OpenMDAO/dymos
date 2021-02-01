import numpy as np
from ....utils.misc import get_rate_units
from ....options import options as dymos_options
import openmdao.api as om


class StateRateCollectorComp(om.ExplicitComponent):
    """
    Class definition for StateRateCollectorComp.

    Collects the state rates and outputs them in the units specified in the state options.
    For explicit integration this is necessary when the output providing the state rate has
    different units than those defined in the state_options/time_options.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
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

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_names = {}

        self._no_check_partials = not dymos_options['include_check_partials']

    def setup(self):
        """
        Create inputs/outputs on this component.
        """
        state_options = self.options['state_options']
        time_units = self.options['time_units']

        for name, options in state_options.items():
            self._input_names[name] = 'state_rates_in:{0}_rate'.format(name)
            self._output_names[name] = 'state_rates:{0}_rate'.format(name)
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_input(self._input_names[name], val=np.ones(shape), units=rate_units)
            self.add_output(self._output_names[name], shape=shape, units=rate_units)

    def compute(self, inputs, outputs):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        state_options = self.options['state_options']

        for name, options in state_options.items():
            outputs[self._output_names[name]] = inputs[self._input_names[name]]
