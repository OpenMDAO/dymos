from __future__ import print_function, division, absolute_import

from six import iteritems
import numpy as np

from openmdao.api import ExplicitComponent, OptionsDictionary

from ...utils.misc import get_rate_units


class SimulationStateMuxComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')
        self.options.declare('times_per_seg', desc='number of points collected per segment')
        # self.options.declare('time_options', types=OptionsDictionary)
        self.options.declare('state_options', types=dict)
        # self.options.declare('control_options', types=dict)
        # self.options.declare('design_parameter_options', types=dict)
        # self.options.declare('input_parameter_options', types=dict)

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        num_points = sum([len(self.options['times_per_seg'][i]) for i in range(num_seg)])

        self._vars = {}

        for name, options in iteritems(self.options['state_options']):
            self._vars[name] = {'inputs': {},
                                'output': 'states:{0}'.format(name),
                                'shape': {}}

            for i in range(num_seg):
                self._vars[name]['inputs'][i] = 'segment_{0}_states:{1}'.format(i, name)
                self._vars[name]['shape'][i] = (len(self.options['times_per_seg'][i]),) + \
                    options['shape']

                self.add_input(name=self._vars[name]['inputs'][i],
                               val=np.ones(self._vars[name]['shape'][i]),
                               units=options['units'])

            self.add_output(name=self._vars[name]['output'],
                            val=np.ones((num_points,) + options['shape']),
                            units=options['units'])


        # for (name, kwargs) in self._timeseries_outputs:
        #
        #     input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
        #     input_name = 'all_values:{0}'.format(name)
        #     self.add_input(input_name,
        #                    shape=(num_nodes,) + kwargs['shape'],
        #                    **input_kwargs)
        #
        #     output_name = name
        #     output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
        #     output_kwargs['shape'] = (num_nodes,) + kwargs['shape']
        #     self.add_output(output_name, **output_kwargs)
        #
        #     self._vars.append((input_name, output_name, kwargs['shape']))
        #
        #     # Setup partials
        #
        #     all_shape = (num_nodes,) + kwargs['shape']
        #     var_size = np.prod(kwargs['shape'])
        #     all_size = np.prod(all_shape)
        #
        #     all_row_starts = gd.subset_node_indices['all'] * var_size
        #     all_rows = []
        #     for i in all_row_starts:
        #         all_rows.extend(range(i, i + var_size))
        #     all_rows = np.asarray(all_rows, dtype=int)
        #
        #     self.declare_partials(
        #         of=output_name,
        #         wrt=input_name,
        #         dependent=True,
        #         rows=all_rows,
        #         cols=np.arange(all_size),
        #         val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for name in self._vars:
            input_names = self._vars[name]['inputs']
            output_name = self._vars[name]['output']

            outputs[output_name] = np.concatenate([inputs[input_names[i]] for i in range(len(input_names))])