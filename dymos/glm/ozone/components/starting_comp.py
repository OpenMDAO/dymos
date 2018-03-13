import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from dymos.glm.ozone.utils.var_names import get_name


class StartingComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)
        self.metadata.declare('num_step_vars', types=int)

    def setup(self):
        num_step_vars = self.metadata['num_step_vars']

        self.declare_partials('*', '*', dependent=False)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            initial_condition_name = get_name('initial_condition', state_name)
            starting_name = get_name('starting', state_name)

            self.add_input(initial_condition_name, shape=state['shape'], units=state['units'])
            self.add_output(starting_name, shape=(num_step_vars,) + state['shape'], units=state['units'])

            ones = np.ones(size)
            arange = np.arange(size)
            self.declare_partials(starting_name, initial_condition_name, val=ones, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for state_name, state in iteritems(self.metadata['states']):
            initial_condition_name = get_name('initial_condition', state_name)
            starting_name = get_name('starting', state_name)

            outputs[starting_name] = 0.
            outputs[starting_name][0, :] = inputs[initial_condition_name]
