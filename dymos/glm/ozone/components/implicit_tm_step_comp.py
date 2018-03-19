import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from dymos.glm.ozone.utils.var_names import get_name
from dymos.glm.ozone.utils.units import get_rate_units


class ImplicitTMStepComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)
        self.metadata.declare('time_units', types=str, allow_none=True)
        self.metadata.declare('num_stages', types=int)
        self.metadata.declare('num_step_vars', types=int)
        self.metadata.declare('glm_B', types=np.ndarray)
        self.metadata.declare('glm_V', types=np.ndarray)
        self.metadata.declare('i_step', types=int)

    def setup(self):
        time_units = self.metadata['time_units']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_step = self.metadata['i_step']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        self.dy_dF = dy_dF = {}

        self.add_input('h', units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name, i_step=i_step)
            y_old_name = get_name('y_old', state_name, i_step=i_step)
            y_new_name = get_name('y_new', state_name, i_step=i_step)

            self.add_input(
                F_name, shape=(num_stages,) + shape,
                units=get_rate_units(state['units'], time_units))

            self.add_input(
                y_old_name, shape=(num_step_vars,) + shape,
                units=state['units'])

            self.add_output(
                y_new_name, shape=(num_step_vars,) + shape,
                units=state['units'])

            F_arange = np.arange(num_stages * size).reshape(
                (num_stages,) + shape)

            y_arange = np.arange(num_step_vars * size).reshape(
                (num_step_vars,) + shape)

            # -----------------

            # (num_step_vars, num_stages,) + shape
            rows = np.einsum('i...,j->ij...', y_arange, np.ones(num_stages, int)).flatten()

            cols = np.zeros((num_step_vars, num_stages,) + shape, int).flatten()
            self.declare_partials(y_new_name, 'h', rows=rows, cols=cols)

            cols = np.einsum('j...,i->ij...', F_arange, np.ones(num_step_vars, int)).flatten()
            self.declare_partials(y_new_name, F_name, rows=rows, cols=cols)

            # -----------------

            # (num_step_vars, num_step_vars,) + shape
            data = np.einsum('ij,...->ij...', glm_V, np.ones(shape)).flatten()
            rows = np.einsum('i...,j->ij...', y_arange, np.ones(num_step_vars)).flatten()
            cols = np.einsum('j...,i->ij...', y_arange, np.ones(num_step_vars)).flatten()

            self.declare_partials(y_new_name, y_old_name, val=data, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']
        i_step = self.metadata['i_step']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name, i_step=i_step)
            y_old_name = get_name('y_old', state_name, i_step=i_step)
            y_new_name = get_name('y_new', state_name, i_step=i_step)

            outputs[y_new_name] = 0. \
                + np.einsum('ij,j...->i...', glm_B, inputs[F_name]) * inputs['h'] \
                + np.einsum('ij,j...->i...', glm_V, inputs[y_old_name])

    def compute_partials(self, inputs, partials):
        time_units = self.metadata['time_units']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']
        i_step = self.metadata['i_step']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name, i_step=i_step)
            y_old_name = get_name('y_old', state_name, i_step=i_step)
            y_new_name = get_name('y_new', state_name, i_step=i_step)

            # (num_stages, num_stages,) + shape

            partials[y_new_name, F_name] = np.einsum(
                '...,ij->ij...', np.ones(shape), glm_B).flatten() * inputs['h']

            partials[y_new_name, 'h'] = np.einsum(
                'ij,j...->ij...', glm_B, inputs[F_name]).flatten()
