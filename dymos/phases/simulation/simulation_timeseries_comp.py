from __future__ import print_function, division, absolute_import

from ..components.timeseries_output_comp import TimeseriesOutputCompBase


class SimulationTimeseriesOutputComp(TimeseriesOutputCompBase):
    """
    The SimulationTimeseriesOutputComp collects simulation data from various sources and
    outputs such that timeseries data can be accessed from SimulationPhase in a way that is
    identical to the other phase classes.
    """
    def initialize(self):
        super(SimulationTimeseriesOutputComp, self).initialize()
        self.options.declare('num_times', desc='Number of time points at which output is requested')

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        num_points = self.options['num_times']

        for (name, kwargs) in self._timeseries_outputs:

            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            input_name = 'all_values:{0}'.format(name)
            self.add_input(input_name,
                           shape=(num_points,) + kwargs['shape'],
                           **input_kwargs)

            output_name = name
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (num_points,) + kwargs['shape']
            self.add_output(output_name, **output_kwargs)

            self._vars.append((input_name, output_name, kwargs['shape']))

            # Setup partials
            self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = inputs[input_name]
