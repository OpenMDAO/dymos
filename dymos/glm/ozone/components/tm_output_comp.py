import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.utils.options_dictionary import OptionsDictionary

from openmdao.api import ExplicitComponent

from dymos.glm.ozone.utils.var_names import get_name


class TMOutputComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)
        self.metadata.declare('num_starting_times', types=int)
        self.metadata.declare('num_my_times', types=int)
        self.metadata.declare('num_step_vars', types=int)
        self.metadata.declare('starting_coeffs', types=np.ndarray, allow_none=True)

    def setup(self):
        num_starting_times = self.metadata['num_starting_times']
        num_my_times = self.metadata['num_my_times']
        num_step_vars = self.metadata['num_step_vars']
        starting_coeffs = self.metadata['starting_coeffs']

        num_times = num_starting_times + num_my_times - 1

        self.declare_partials('*', '*', dependent=False)

        for state_name, state in iteritems(self.metadata['states']):
            shape = state['shape']
            size = np.prod(shape)

            starting_state_name = get_name('starting_state', state_name)
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)

            for i_step in range(num_my_times):
                y_name = get_name('y', state_name, i_step=i_step)

                self.add_input(
                    y_name,
                    shape=(num_step_vars,) + shape,
                    units=state['units'])

            self.add_output(
                out_state_name,
                shape=(num_times,) + state['shape'],
                units=state['units'])

            y_arange = np.arange(num_step_vars * size).reshape((num_step_vars,) + shape)

            state_arange = np.arange(num_times * size).reshape(
                (num_times,) + shape)

            for i_step in range(num_my_times):
                y_name = get_name('y', state_name, i_step=i_step)

                data = np.ones(size)
                rows = state_arange[i_step + num_starting_times - 1, :]
                cols = y_arange[0, :]

                self.declare_partials(out_state_name, y_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        num_starting_times = self.metadata['num_starting_times']
        num_my_times = self.metadata['num_my_times']
        num_step_vars = self.metadata['num_step_vars']
        starting_coeffs = self.metadata['starting_coeffs']

        for state_name, state in iteritems(self.metadata['states']):
            starting_state_name = get_name('starting_state', state_name)
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)

            for i_step in range(num_my_times):
                y_name = get_name('y', state_name, i_step=i_step)

                outputs[out_state_name][i_step + num_starting_times - 1, :] = \
                    inputs[y_name][0, :]
