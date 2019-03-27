from __future__ import print_function, division, absolute_import

import numpy as np

from ...common.timeseries_output_comp import TimeseriesOutputCompBase


class RungeKuttaTimeseriesOutputComp(TimeseriesOutputCompBase):

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        grid_data = self.options['grid_data']
        num_nodes = 2 * grid_data.num_segments

        for (name, kwargs) in self._timeseries_outputs:

            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            input_name = 'segend_values:{0}'.format(name)
            self.add_input(input_name,
                           shape=(num_nodes,) + kwargs['shape'],
                           **input_kwargs)

            output_name = name
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (num_nodes,) + kwargs['shape']
            self.add_output(output_name, **output_kwargs)

            self._vars.append((input_name, output_name, kwargs['shape']))

            # Setup partials
            segend_shape = (num_nodes,) + kwargs['shape']
            segend_size = np.prod(segend_shape)

            ar = np.arange(segend_size)

            self.declare_partials(
                of=output_name,
                wrt=input_name,
                rows=ar,
                cols=ar,
                val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = inputs[input_name]
