import numpy as np
from six import iteritems
import scipy.sparse

from openmdao.api import ExplicitComponent

from dymos.glm.ozone.utils.var_names import get_name
from dymos.glm.ozone.utils.units import get_rate_units


class ExplicitTMStepComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('states', types=dict)
        self.options.declare('time_units', types=str, allow_none=True)
        self.options.declare('num_stages', types=int)
        self.options.declare('num_step_vars', types=int)
        self.options.declare('glm_B', types=np.ndarray)
        self.options.declare('glm_V', types=np.ndarray)
        self.options.declare('i_step', types=int)

    def setup(self):
        time_units = self.options['time_units']
        num_stages = self.options['num_stages']
        num_step_vars = self.options['num_step_vars']
        i_step = self.options['i_step']
        glm_B = self.options['glm_B']
        glm_V = self.options['glm_V']

        self.dy_dF = dy_dF = {}

        self.declare_partials('*', '*', dependent=False)

        self.add_input('h', units=time_units)

        for state_name, state in iteritems(self.options['states']):
            size = np.prod(state['shape'])

            y_old_name = get_name('y_old', state_name, i_step=i_step)
            y_new_name = get_name('y_new', state_name, i_step=i_step)

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                self.add_input(
                    F_name, shape=(1,) + state['shape'],
                    units=get_rate_units(state['units'], time_units))

            self.add_input(
                y_old_name, shape=(num_step_vars,) + state['shape'],
                units=state['units'])

            self.add_output(
                y_new_name, shape=(num_step_vars,) + state['shape'],
                units=state['units'])

            self.declare_partials(y_new_name, 'h', dependent=True)

            y_arange = np.arange(num_step_vars * size).reshape((num_step_vars, size))

            # num_step_vars, num_step_vars, size
            data = np.einsum('ij,...->ij...', glm_V, np.ones(size)).flatten()
            rows = np.einsum('i...,j->ij...', y_arange, np.ones(num_step_vars, int)).flatten()
            cols = np.einsum('j...,i->ij...', y_arange, np.ones(num_step_vars, int)).flatten()

            self.declare_partials(y_new_name, y_old_name, val=data, rows=rows, cols=cols)

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                vals = np.zeros((num_step_vars, 1, size))
                rows = np.arange(num_step_vars * 1 * size)
                cols = np.zeros((num_step_vars, 1 * size), int)
                for ii_step in range(num_step_vars):
                    vals[ii_step, 0, :] = glm_B[ii_step, j_stage]
                    cols[ii_step, :] = np.arange(1 * size)
                vals = vals.flatten()
                cols = cols.flatten()

                self.declare_partials(y_new_name, F_name, rows=rows, cols=cols)
                dy_dF[state_name, j_stage] = vals

    def compute(self, inputs, outputs):
        num_stages = self.options['num_stages']
        num_step_vars = self.options['num_step_vars']
        glm_B = self.options['glm_B']
        glm_V = self.options['glm_V']
        i_step = self.options['i_step']

        ii_step = 0

        for state_name, state in iteritems(self.options['states']):
            size = np.prod(state['shape'])

            y_old_name = get_name('y_old', state_name, i_step=i_step)
            y_new_name = get_name('y_new', state_name, i_step=i_step)

            outputs[y_new_name] = np.einsum('ij,jk...->ik...', glm_V, inputs[y_old_name])

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                # y_new:step x shape
                # GLM step x stage
                # F: 1 x shape
                outputs[y_new_name] += inputs['h'] * np.einsum(
                    'i,...->i...', glm_B[:, j_stage], inputs[F_name][0, :])

    def compute_partials(self, inputs, partials):
        num_stages = self.options['num_stages']
        num_step_vars = self.options['num_step_vars']
        glm_B = self.options['glm_B']
        glm_V = self.options['glm_V']
        i_step = self.options['i_step']

        ii_step = 0

        dy_dF = self.dy_dF

        for state_name, state in iteritems(self.options['states']):
            size = np.prod(state['shape'])

            y_new_name = get_name('y_new', state_name, i_step=i_step)

            partials[y_new_name, 'h'][:, 0] = 0.

            for j_stage in range(num_stages):
                F_name = get_name('F', state_name, i_step=i_step, j_stage=j_stage)

                partials[y_new_name, F_name] = inputs['h'] * dy_dF[state_name, j_stage]

                # partials[y_new_name, 'h'][:, 0] += glm_B[ii_step, j_stage] \
                #     * inputs[F_name].flatten()

                partials[y_new_name, 'h'][:, 0] += np.einsum(
                    'i,...->i...', glm_B[:, j_stage], inputs[F_name][0, :]).flatten()
