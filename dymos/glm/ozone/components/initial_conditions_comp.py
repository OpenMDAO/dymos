import numpy as np
from six import iteritems

from openmdao.api import ExplicitComponent


class InitialConditionsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)

    def setup(self):
        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            in_name = 'in:%s' % state_name
            out_name = 'out:%s' % state_name

            self.add_input(in_name, shape=shape, units=state['units'])
            self.add_output(out_name, shape=shape, units=state['units'])

            arange = np.arange(size)
            self.declare_partials(out_name, in_name, val=1., rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for state_name, state in iteritems(self.metadata['states']):
            in_name = 'in:%s' % state_name
            out_name = 'out:%s' % state_name

            outputs[out_name] = inputs[in_name]
