import numpy as np
from six import iteritems
import scipy.sparse
import scipy.sparse.linalg

from openmdao.api import ImplicitComponent

from dymos.glm.ozone.utils.var_names import get_name
from dymos.glm.ozone.utils.units import get_rate_units


class VectorizedStepComp(ImplicitComponent):

    def initialize(self):
        self.metadata.declare('states', types=dict)
        self.metadata.declare('time_units', types=str, allow_none=True)
        self.metadata.declare('num_times', types=int)
        self.metadata.declare('num_stages', types=int)
        self.metadata.declare('num_step_vars', types=int)
        self.metadata.declare('glm_B', types=np.ndarray)
        self.metadata.declare('glm_V', types=np.ndarray)

    def setup(self):
        time_units = self.metadata['time_units']
        num_times = self.metadata['num_times']
        num_stages = self.metadata['num_stages']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']
        glm_V = self.metadata['glm_V']

        self.dy_dy = dy_dy = {}
        self.dy_dy_inv = dy_dy_inv = {}

        h_arange = np.arange(num_times - 1)

        self.add_input('h_vec', shape=(num_times - 1), units=time_units)

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            y0_name = get_name('y0', state_name)
            y_name = get_name('y', state_name)

            y0_arange = np.arange(num_step_vars * size).reshape((num_step_vars,) + shape)

            y_arange = np.arange(num_times * num_step_vars * size).reshape(
                (num_times, num_step_vars,) + shape)

            F_arange = np.arange((num_times - 1) * num_stages * size).reshape(
                (num_times - 1, num_stages,) + shape)

            self.add_input(F_name,
                shape=(num_times - 1, num_stages,) + shape,
                units=get_rate_units(state['units'], time_units))

            self.add_input(y0_name,
                shape=(num_step_vars,) + shape,
                units=state['units'])

            self.add_output(y_name,
                shape=(num_times, num_step_vars,) + shape,
                units=state['units'])

            # -----------------

            # (num_times, num_step_vars,) + shape
            data1 = np.ones(num_times * num_step_vars * size)
            rows1 = np.arange(num_times * num_step_vars * size)
            cols1 = np.arange(num_times * num_step_vars * size)

            # (num_times - 1, num_step_vars, num_step_vars,) + shape
            data2 = np.einsum('i...,jk->ijk...',
                np.ones((num_times - 1,) + shape), -glm_V).flatten()
            rows2 = np.einsum('ij...,k->ijk...',
                y_arange[1:, :, :], np.ones(num_step_vars)).flatten()
            cols2 = np.einsum('ik...,j->ijk...',
                y_arange[:-1, :, :], np.ones(num_step_vars)).flatten()

            data = np.concatenate([data1, data2])
            rows = np.concatenate([rows1, rows2])
            cols = np.concatenate([cols1, cols2])

            dy_dy[state_name] = scipy.sparse.csc_matrix(
                (data, (rows, cols)),
                shape=(
                    num_times * num_step_vars * size,
                    num_times * num_step_vars * size))

            dy_dy_inv[state_name] = scipy.sparse.linalg.splu(dy_dy[state_name])

            self.declare_partials(y_name, y_name, val=data, rows=rows, cols=cols)

            # -----------------

            # (num_step_vars,) + shape
            data = -np.ones((num_step_vars,) + shape).flatten()
            rows = y_arange[0, :, :].flatten()
            cols = y0_arange.flatten()

            self.declare_partials(y_name, y0_name, val=data, rows=rows, cols=cols)

            # -----------------

            # (num_times - 1, num_step_vars, num_stages,) + shape
            rows = np.einsum('ij...,k->ijk...', y_arange[1:, :, :], np.ones(num_stages)).flatten()

            cols = np.einsum('jk...,i->ijk...',
                np.ones((num_step_vars, num_stages,) + shape), h_arange).flatten()
            self.declare_partials(y_name, 'h_vec', rows=rows, cols=cols)

            cols = np.einsum('ik...,j->ijk...', F_arange, np.ones(num_step_vars)).flatten()
            self.declare_partials(y_name, F_name, rows=rows, cols=cols)

    def apply_nonlinear(self, inputs, outputs, residuals):
        num_times = self.metadata['num_times']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']

        dy_dy = self.dy_dy

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            y0_name = get_name('y0', state_name)
            y_name = get_name('y', state_name)

            # dy_dy term
            in_vec = outputs[y_name].reshape((num_times * num_step_vars * size))
            out_vec = dy_dy[state_name].dot(in_vec).reshape(
                (num_times, num_step_vars,) + shape)

            residuals[y_name] = out_vec # y term
            residuals[y_name][0, :, :] -= inputs[y0_name] # y0 term
            residuals[y_name][1:, :, :] -= np.einsum('jl,i,il...->ij...',
                glm_B, inputs['h_vec'], inputs[F_name]) # hF term

    def solve_nonlinear(self, inputs, outputs):
        num_times = self.metadata['num_times']
        num_step_vars = self.metadata['num_step_vars']
        glm_B = self.metadata['glm_B']

        dy_dy_inv = self.dy_dy_inv

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            y0_name = get_name('y0', state_name)
            y_name = get_name('y', state_name)

            vec = np.zeros((num_times, num_step_vars,) + shape)
            vec[0, :, :] += inputs[y0_name] # y0 term
            vec[1:, :, :] += np.einsum('jl,i,il...->ij...',
                glm_B, inputs['h_vec'], inputs[F_name]) # hF term

            outputs[y_name] = dy_dy_inv[state_name].solve(vec.flatten(), 'N').reshape(
                (num_times, num_step_vars,) + shape)

    def linearize(self, inputs, outputs, partials):
        glm_B = self.metadata['glm_B']

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            F_name = get_name('F', state_name)
            y0_name = get_name('y0', state_name)
            y_name = get_name('y', state_name)

            # (num_times - 1, num_step_vars, num_stages,) + shape

            partials[y_name, F_name] = -np.einsum(
                '...,jk,i->ijk...', np.ones(shape), glm_B, inputs['h_vec']).flatten()

            partials[y_name, 'h_vec'] = -np.einsum(
                'jk,ik...->ijk...', glm_B, inputs[F_name]).flatten()

    def solve_linear(self, d_outputs, d_residuals, mode):
        num_times = self.metadata['num_times']
        num_step_vars = self.metadata['num_step_vars']

        dy_dy_inv = self.dy_dy_inv

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y_name = get_name('y', state_name)

            if mode == 'fwd':
                rhs_vec = d_residuals[y_name].flatten()
                solve_mode = 'N'
            elif mode == 'rev':
                rhs_vec = d_outputs[y_name].flatten()
                solve_mode = 'T'

            sol_vec = dy_dy_inv[state_name].solve(rhs_vec, solve_mode)

            if mode == 'fwd':
                d_outputs[y_name] = sol_vec.reshape((num_times, num_step_vars,) + shape)
            elif mode == 'rev':
                d_residuals[y_name] = sol_vec.reshape((num_times, num_step_vars,) + shape)

    def solve_multi_linear(self, d_outputs, d_residuals, mode):
        num_times = self.metadata['num_times']
        num_step_vars = self.metadata['num_step_vars']

        dy_dy_inv = self.dy_dy_inv

        for state_name, state in iteritems(self.metadata['states']):
            size = np.prod(state['shape'])
            shape = state['shape']

            y_name = get_name('y', state_name)

            nrow = num_times * num_step_vars * size
            ncol = d_outputs[y_name].shape[-1]

            if mode == 'fwd':
                rhs_array = d_residuals[y_name].reshape((nrow, ncol))
                sol_array = d_outputs[y_name].reshape((nrow, ncol))
                solve_mode = 'N'
            elif mode == 'rev':
                rhs_array = d_outputs[y_name].reshape((nrow, ncol))
                sol_array = d_residuals[y_name].reshape((nrow, ncol))
                solve_mode = 'T'

            for icol in range(ncol):
                rhs = rhs_array[:, icol]
                sol = dy_dy_inv[state_name].solve(rhs, solve_mode)
                sol_array[:, icol] = sol
