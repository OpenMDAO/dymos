from __future__ import print_function, division, absolute_import

from six import iteritems
import numpy as np

from openmdao.api import ExplicitComponent


class SegmentStateMuxComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('grid_data', desc='the grid data of the corresponding phase.')
        self.options.declare('state_options', types=dict)

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        num_nodes = gd.subset_num_nodes['all']

        self._vars = {}

        for name, options in iteritems(self.options['state_options']):
            self._vars[name] = {'inputs': {},
                                'output': 'states:{0}'.format(name),
                                'shape': {}}

            for i in range(num_seg):
                nnps_i = gd.subset_num_nodes_per_segment['all'][i]
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
