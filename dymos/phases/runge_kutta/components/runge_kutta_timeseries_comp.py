from __future__ import print_function, division, absolute_import

import numpy as np

from ...components.timeseries_output_comp import TimeseriesOutputCompBase


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
            all_shape = (num_nodes,) + kwargs['shape']
            var_size = np.prod(kwargs['shape'])
            all_size = np.prod(all_shape)

            all_row_starts = grid_data.subset_node_indices['all'] * var_size
            all_rows = []
            for i in all_row_starts:
                all_rows.extend(range(i, i + var_size))
            all_rows = np.asarray(all_rows, dtype=int)

            # self.declare_partials(
            #     of=output_name,
            #     wrt=input_name,
            #     dependent=True,
            #     rows=all_rows,
            #     cols=np.arange(all_size),
            #     val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = inputs[input_name]
