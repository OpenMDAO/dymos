import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from dymos.glm.ozone.utils.var_names import get_name
from dymos.glm.ozone.utils.units import get_rate_units


class ExplicitTMStageComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)
        self.metadata.declare('time_units', types=str, allow_none=True)
        self.metadata.declare('num_stages', types=int)
        self.metadata.declare('num_step_vars', types=int)
        self.metadata.declare('glm_A', types=np.ndarray)
        self.metadata.declare('glm_U', types=np.ndarray)
        self.metadata.declare('i_stage', types=int)
        self.metadata.declare('i_step', types=int)

    def setup(self):
        time_units = self.metadata['time_units']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_stage = self.metadata['i_stage']
        i_step = self.metadata['i_step']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']

        self.declare_partials('*', '*', dependent=False)

        self.add_input('h', units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_old_name = get_name('y_old', state_name, i_step=i_step, i_stage=i_stage)
            Y_name = get_name('Y', state_name, i_step=i_step, i_stage=i_stage)

            for j_stage in range(i_stage):
                F_name = get_name('F', state_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage)

                self.add_input(F_name, shape=(1,) + state['shape'],
                    units=get_rate_units(state['units'], time_units))

            self.add_input(y_old_name, shape=(num_step_vars,) + state['shape'],
                units=state['units'])

            self.add_output(Y_name, shape=(1,) + state['shape'],
                units=state['units'])

            vals = np.zeros((num_step_vars, size))
            rows = np.zeros((num_step_vars, size), int)
            cols = np.arange(num_step_vars * size)
            for ii_step in range(num_step_vars):
                vals[ii_step, :] = glm_U[i_stage, ii_step]
                rows[ii_step, :] = np.arange(size)
            vals = vals.flatten()
            rows = rows.flatten()

            self.declare_partials(Y_name, 'h', dependent=True)

            self.declare_partials(Y_name, y_old_name, val=vals, rows=rows, cols=cols)

            for j_stage in range(i_stage):
                F_name = get_name('F', state_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage)

                arange = np.arange(size)
                self.declare_partials(Y_name, F_name, rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_stage = self.metadata['i_stage']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']
        i_step = self.metadata['i_step']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            y_old_name = get_name('y_old', state_name, i_step=i_step, i_stage=i_stage)
            Y_name = get_name('Y', state_name, i_step=i_step, i_stage=i_stage)

            outputs[Y_name][0, :] = np.einsum('i,i...->...', glm_U[i_stage, :], inputs[y_old_name])

            for j_stage in range(i_stage):
                F_name = get_name('F', state_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage)

                outputs[Y_name] += inputs['h'] * glm_A[i_stage, j_stage] * inputs[F_name]

    def compute_partials(self, inputs, partials):
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        i_stage = self.metadata['i_stage']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']
        i_step = self.metadata['i_step']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])

            Y_name = get_name('Y', state_name, i_step=i_step, i_stage=i_stage)

            partials[Y_name, 'h'][:, 0] = 0.

            for j_stage in range(i_stage):
                F_name = get_name('F', state_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage)

                partials[Y_name, F_name] = inputs['h'] * glm_A[i_stage, j_stage]

                partials[Y_name, 'h'][:, 0] += glm_A[i_stage, j_stage] * inputs[F_name].flatten()
