import numpy as np
from six import iteritems
import scipy.sparse
import scipy.sparse.linalg

from openmdao.api import ExplicitComponent

from dymos.glm.ozone.utils.var_names import get_name
from dymos.glm.ozone.utils.units import get_rate_units


class VectorizedStageComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)
        self.metadata.declare('time_units', types=str, allow_none=True)
        self.metadata.declare('num_times', types=int)
        self.metadata.declare('num_stages', types=int)
        self.metadata.declare('num_step_vars', types=int)
        self.metadata.declare('glm_A', types=np.ndarray)
        self.metadata.declare('glm_U', types=np.ndarray)

    def setup(self):
        time_units = self.metadata['time_units']
        num_times = self.metadata['num_times']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        h_arange = np.arange(num_times - 1)

        self.add_input('h_vec', shape=(num_times - 1), units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            y_name = get_name('y', state_name)
            Y_out_name = get_name('Y_out', state_name)
            Y_in_name = get_name('Y_in', state_name)

            Y_arange = np.arange((num_times - 1) * num_stages * size).reshape(
                (num_times - 1, num_stages,) + shape)

            F_arange = np.arange((num_times - 1) * num_stages * size).reshape(
                (num_times - 1, num_stages,) + shape)

            y_arange = np.arange(num_times * num_step_vars * size).reshape(
                (num_times, num_step_vars,) + shape)

            self.add_input(
                F_name,
                shape=(num_times - 1, num_stages,) + shape,
                units=get_rate_units(state['units'], time_units))

            self.add_input(
                y_name,
                shape=(num_times, num_step_vars,) + shape,
                units=state['units'])

            self.add_input(
                Y_in_name, val=0.,
                shape=(num_times - 1, num_stages,) + shape,
                units=state['units'])

            self.add_output(
                Y_out_name,
                shape=(num_times - 1, num_stages,) + shape,
                units=state['units'])

            # -----------------

            ones = -np.ones((num_times - 1) * num_stages * size)
            arange = np.arange((num_times - 1) * num_stages * size)
            self.declare_partials(Y_out_name, Y_in_name, val=ones, rows=arange, cols=arange)

            # -----------------

            # (num_times - 1, num_stages, num_stages,) + shape
            rows = np.einsum('ij...,k->ijk...', Y_arange, np.ones(num_stages, int)).flatten()

            cols = np.einsum(
                'jk...,i->ijk...',
                np.ones((num_stages, num_stages,) + shape, int), h_arange).flatten()
            self.declare_partials(Y_out_name, 'h_vec', rows=rows, cols=cols)

            cols = np.einsum('ik...,j->ijk...', F_arange, np.ones(num_stages, int)).flatten()
            self.declare_partials(Y_out_name, F_name, rows=rows, cols=cols)

            # -----------------

            # (num_times - 1, num_stages, num_step_vars,) + shape
            data = np.einsum(
                'jk,i...->ijk...', glm_U, np.ones((num_times - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...', Y_arange, np.ones(num_step_vars)).flatten()
            cols = np.einsum('ik...,j->ijk...', y_arange[:-1, :, :], np.ones(num_stages)).flatten()

            self.declare_partials(Y_out_name, y_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            y_name = get_name('y', state_name)
            Y_out_name = get_name('Y_out', state_name)
            Y_in_name = get_name('Y_in', state_name)

            outputs[Y_out_name] = -inputs[Y_in_name] \
                + np.einsum('jk,i,ik...->ij...', glm_A, inputs['h_vec'], inputs[F_name]) \
                + np.einsum('jk,ik...->ij...', glm_U, inputs[y_name][:-1, :, :])

    def compute_partials(self, inputs, partials):
        time_units = self.metadata['time_units']
        num_times = self.metadata['num_times']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            y_name = get_name('y', state_name)
            Y_out_name = get_name('Y_out', state_name)
            Y_in_name = get_name('Y_in', state_name)

            # (num_times - 1, num_stages, num_stages,) + shape

            partials[Y_out_name, F_name] = np.einsum(
                '...,jk,i->ijk...', np.ones(shape), glm_A, inputs['h_vec']).flatten()

            partials[Y_out_name, 'h_vec'] = np.einsum(
                'jk,ik...->ijk...', glm_A, inputs[F_name]).flatten()
