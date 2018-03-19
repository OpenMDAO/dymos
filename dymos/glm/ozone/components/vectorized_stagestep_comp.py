import numpy as np
from six import iteritems
import scipy.sparse
import scipy.sparse.linalg

from openmdao.api import ExplicitComponent

from dymos.glm.ozone.utils.var_names import get_name
from dymos.glm.ozone.utils.units import get_rate_units


class VectorizedStageStepComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)
        self.metadata.declare('time_units', types=str, allow_none=True)
        self.metadata.declare('num_times', types=int)
        self.metadata.declare('num_stages', types=int)
        self.metadata.declare('num_step_vars', types=int)
        self.metadata.declare('glm_A', types=np.ndarray)
        self.metadata.declare('glm_U', types=np.ndarray)
        self.metadata.declare('glm_B', types=np.ndarray)
        self.metadata.declare('glm_V', types=np.ndarray)

    def setup(self):
        time_units = self.metadata['time_units']
        num_times = self.metadata['num_times']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_A = self.metadata['glm_A']
        glm_U = self.metadata['glm_U']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        self.mtx_y0_dict = {}
        self.mtx_dict = {}
        self.mtx_h_dict = {}

        h_arange = np.arange(num_times - 1)
        num_h = num_times - 1

        self.add_input('h_vec', shape=(num_times - 1), units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y0_name = get_name('y0', state_name)
            F_name = get_name('F', state_name)
            Y_in_name = get_name('Y_in', state_name)
            Y_out_name = get_name('Y_out', state_name)

            # --------------------------------------------------------------------------------

            y0_arange = np.arange(num_step_vars * size).reshape((num_step_vars,) + shape)

            F_arange = np.arange((num_times - 1) * num_stages * size).reshape(
                (num_times - 1, num_stages,) + shape)

            Y_arange = np.arange((num_times - 1) * num_stages * size).reshape(
                (num_times - 1, num_stages,) + shape)

            y_arange = np.arange(num_times * num_step_vars * size).reshape(
                (num_times, num_step_vars,) + shape)

            num_y0 = np.prod(y0_arange.shape)
            num_F = np.prod(F_arange.shape)
            num_Y = np.prod(Y_arange.shape)
            num_y = np.prod(y_arange.shape)

            # --------------------------------------------------------------------------------

            self.add_input(
                y0_name,
                shape=(num_step_vars,) + shape,
                units=state['units'])

            self.add_input(
                F_name,
                shape=(num_times - 1, num_stages,) + shape,
                units=get_rate_units(state['units'], time_units))

            self.add_input(
                Y_in_name, val=0.,
                shape=(num_times - 1, num_stages,) + shape,
                units=state['units'])

            self.add_output(
                Y_out_name,
                shape=(num_times - 1, num_stages,) + shape,
                units=state['units'])

            # -----------------

            self.declare_partials(Y_out_name, 'h_vec')
            self.declare_partials(Y_out_name, y0_name)
            self.declare_partials(Y_out_name, F_name)

            # -----------------

            ones = -np.ones((num_times - 1) * num_stages * size)
            arange = np.arange((num_times - 1) * num_stages * size)
            self.declare_partials(Y_out_name, Y_in_name, val=ones, rows=arange, cols=arange)

            # --------------------------------------------------------------------------------
            # mtx_y0: num_stages x num_step_vars x ...

            data = np.ones((num_step_vars,) + shape).flatten()
            rows = y_arange[0, :, :].flatten()
            cols = y0_arange.flatten()
            mtx_y0 = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(num_y, num_y0)).toarray()

            # --------------------------------------------------------------------------------
            # mtx_A: (num_times - 1) x num_stages x num_stages x ...

            data = np.einsum('jk,i...->ijk...', glm_A, np.ones((num_times - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...', Y_arange, np.ones(num_stages, int)).flatten()
            cols = np.einsum('ik...,j->ijk...', F_arange, np.ones(num_stages, int)).flatten()
            mtx_A = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(num_Y, num_F)).toarray()

            # --------------------------------------------------------------------------------
            # mtx_B: (num_times - 1) x num_step_vars x num_stages x ...

            data = np.einsum('jk,i...->ijk...', glm_B, np.ones((num_times - 1,) + shape)).flatten()
            rows = np.einsum(
                'ij...,k->ijk...',
                y_arange[1:, :, :], np.ones(num_stages, int)).flatten()
            cols = np.einsum('ik...,j->ijk...', F_arange, np.ones(num_step_vars, int)).flatten()
            mtx_B = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(num_y, num_F)).toarray()

            # --------------------------------------------------------------------------------
            # mtx_U: (num_times - 1) x num_stages x num_step_vars x ...

            data = np.einsum('jk,i...->ijk...', glm_U, np.ones((num_times - 1,) + shape)).flatten()
            rows = np.einsum('ij...,k->ijk...', Y_arange, np.ones(num_step_vars, int)).flatten()
            cols = np.einsum(
                'ik...,j->ijk...',
                y_arange[:-1, :, :], np.ones(num_stages, int)).flatten()
            mtx_U = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(num_Y, num_y)).toarray()

            # --------------------------------------------------------------------------------
            # mtx_y

            data_list = []
            rows_list = []
            cols_list = []

            # identity
            data = np.ones(num_y)
            rows = np.arange(num_y)
            cols = np.arange(num_y)
            data_list.append(data)
            rows_list.append(rows)
            cols_list.append(cols)

            # (num_times - 1) x num_step_var x num_step_var x ...
            data = np.einsum(
                'jk,i...->ijk...',
                -glm_V, np.ones((num_times - 1,) + shape)).flatten()
            rows = np.einsum(
                'ij...,k->ijk...',
                y_arange[1:, :, :], np.ones(num_step_vars, int)).flatten()
            cols = np.einsum(
                'ik...,j->ijk...',
                y_arange[:-1, :, :], np.ones(num_step_vars, int)).flatten()
            data_list.append(data)
            rows_list.append(rows)
            cols_list.append(cols)

            # concatenate
            data = np.concatenate(data_list)
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)

            mtx_y = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(num_y, num_y))
            mtx_y_inv = scipy.sparse.linalg.splu(mtx_y)

            # --------------------------------------------------------------------------------
            # mtx_h

            data = np.ones(num_F)
            rows = np.arange(num_F)
            cols = np.einsum(
                'i,j...->ij...',
                h_arange, np.ones((num_stages,) + shape, int)).flatten()
            mtx_h = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(num_F, num_h)).toarray()

            # --------------------------------------------------------------------------------
            self.mtx_y0_dict[state_name] = mtx_U.dot(mtx_y_inv.solve(mtx_y0))
            self.mtx_dict[state_name] = mtx_A + mtx_U.dot(mtx_y_inv.solve(mtx_B))
            self.mtx_h_dict[state_name] = mtx_h

    def compute(self, inputs, outputs):
        num_times = self.metadata['num_times']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            Y_out_name = get_name('Y_out', state_name)
            Y_in_name = get_name('Y_in', state_name)
            y0_name = get_name('y0', state_name)

            mtx_y0 = self.mtx_y0_dict[state_name]
            mtx = self.mtx_dict[state_name]
            mtx_h = self.mtx_h_dict[state_name]

            Y_shape = outputs[Y_out_name].shape

            outputs[Y_out_name] = -inputs[Y_in_name] \
                + mtx.dot(mtx_h.dot(inputs['h_vec']) * inputs[F_name].flatten()).reshape(Y_shape) \
                + mtx_y0.dot(inputs[y0_name].flatten()).reshape(Y_shape)

    def compute_partials(self, inputs, partials):
        num_times = self.metadata['num_times']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            Y_out_name = get_name('Y_out', state_name)
            Y_in_name = get_name('Y_in', state_name)
            y0_name = get_name('y0', state_name)

            mtx_y0 = self.mtx_y0_dict[state_name]
            mtx = self.mtx_dict[state_name]
            mtx_h = self.mtx_h_dict[state_name]

            partials[Y_out_name, 'h_vec'][:, :] = \
                mtx.dot(np.diag(inputs[F_name].flatten())).dot(mtx_h)
            partials[Y_out_name, F_name][:, :] = mtx.dot(np.diag(mtx_h.dot(inputs['h_vec'])))
            partials[Y_out_name, y0_name][:, :] = mtx_y0
