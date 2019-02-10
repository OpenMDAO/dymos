import numpy as np
from dymos.utils.misc import get_rate_units
from openmdao.core.explicitcomponent import ExplicitComponent
from six import string_types, iteritems


class StateRateCollectorComp(ExplicitComponent):
    """
    Collects the state rates and outputs them in the units specified in the state options.
    For explicit integration this is necessary when the output providing the state rate has
    different units than those defined in the state_options/time_options.
    """
    def initialize(self):
        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of options for the ODE state variables.')
        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_names = {}

    def setup(self):
        state_options = self.options['state_options']
        time_units = self.options['time_units']

        for name, options in iteritems(state_options):
            self._input_names[name] = 'state_rates_in:{0}_rate'.format(name)
            self._output_names[name] = 'state_rates:{0}_rate'.format(name)
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_input(self._input_names[name], val=np.ones(shape), units=rate_units)
            self.add_output(self._output_names[name], shape=shape, units=rate_units)

    def compute(self, inputs, outputs):
        state_options = self.options['state_options']

        for name, options in iteritems(state_options):
            outputs[self._output_names[name]] = inputs[self._input_names[name]]
