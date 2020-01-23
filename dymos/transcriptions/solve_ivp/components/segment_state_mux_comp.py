import numpy as np

import openmdao.api as om


class SegmentStateMuxComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')
        self.options.declare('state_options', types=dict)

        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        gd = self.options['grid_data']
        num_seg = gd.num_segments

        if self.options['output_nodes_per_seg'] is None:
            num_nodes = gd.subset_num_nodes['all']
        else:
            num_nodes = num_seg * self.options['output_nodes_per_seg']

        self._vars = {}

        for name, options in self.options['state_options'].items():
            self._vars[name] = {'inputs': {},
                                'output': 'states:{0}'.format(name),
                                'shape': {}}

            for i in range(num_seg):
                if self.options['output_nodes_per_seg'] is None:
                    nnps_i = gd.subset_num_nodes_per_segment['all'][i]
                else:
                    nnps_i = self.options['output_nodes_per_seg']
                self._vars[name]['inputs'][i] = 'segment_{0}_states:{1}'.format(i, name)
                self._vars[name]['shape'][i] = (nnps_i,) + options['shape']

                self.add_input(name=self._vars[name]['inputs'][i],
                               val=np.ones(self._vars[name]['shape'][i]),
                               units=options['units'])

                self.declare_partials(of=self._vars[name]['output'],
                                      wrt=self._vars[name]['inputs'][i],
                                      method='fd')

            self.add_output(name=self._vars[name]['output'],
                            val=np.ones((num_nodes,) + options['shape']),
                            units=options['units'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for name in self._vars:
            input_names = self._vars[name]['inputs']
            output_name = self._vars[name]['output']

            outputs[output_name] = \
                np.concatenate([inputs[input_names[i]] for i in range(len(input_names))])
