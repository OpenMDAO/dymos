from __future__ import division, print_function, absolute_import

from collections import Iterable, Sequence
from ...transcriptions.grid_data import GridData
from ...phase.phase import Phase
from ...utils.lgr import lgr
from ...utils.lagrange import lagrange_matrices

from scipy.linalg import block_diag

import numpy as np

import openmdao.api as om
from openmdao.api import Problem
from openmdao.core.system import System
import dymos as dm


def err(y_dot, f):
    error = np.absolute(y_dot - f)
    return error


def quadrature(func, t0, t1, n):
    x, w = lgr(n)
    x = 0.5*(x + 1)*(t1 - t0) + t0
    w = 0.5*(w + 1)*(t1 - t0) + t0

    integral = func(x[0])*w[0]

    for i in range(1, n+1):
        integral += func(x[i])*w[i]

    return integral


class PHAdaptive():

    def __init__(self, phase, tol=1e-6, min_order=3, max_order=7):
        self.phase = phase
        self.tol = tol
        self.min_order = min_order
        self.max_order = max_order
        self.gd = phase.options['transcription'].grid_data
        self.error = 0

    def check_error(self):
        gd = self.gd
        phase = self.phase
        num_nodes = gd.subset_num_nodes['all']
        numseg = gd.num_segments

        inputs = phase.list_inputs(units=False, out_stream=None)
        outputs = phase.list_outputs(units=False, out_stream=None)

        in_values_dict = {k: v['value'] for k, v in inputs}
        out_values_dict = {k: v['value'] for k, v in outputs}

        prom_to_abs_map = phase._var_allprocs_prom2abs_list['output']

        print(out_values_dict.keys())

        num_scalar_states = 0
        for state_name, options in phase.state_options.items():
            shape = options['shape']
            size = np.prod(shape)
            num_scalar_states += size

        y = np.zeros([num_nodes, num_scalar_states])
        f = np.zeros([num_nodes, num_scalar_states])
        c = 0
        for state_name, options in phase.state_options.items():
            prom_name = f'timeseries.states:{state_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            rate_source_prom_name = 'rhs_all.' + options['rate_source']
            rate_abs_name = prom_to_abs_map[rate_source_prom_name][0]
            y[:, c] = out_values_dict[abs_name].ravel()
            f[:, c] = out_values_dict[rate_abs_name].ravel()
            c += 1

        # interpolate y at t_hat
        new_order = gd.transcription_order + 1
        new_grid = GridData(numseg, gd.transcription, new_order, gd.segment_ends, gd.compressed)

        L = self.interpolation_lagrange_matrix(gd, new_grid)
        I = self.integration_matrix(new_grid)
        y_hat = np.dot(L, y)

        y_prime = y[0, :] + np.dot(I, f)

        E = np.absolute(y_prime - y_hat)
        self.error = E/(1 + np.max(y_hat))

        need_refinement = False
        print(self.error)
        if np.any(self.error > self.tol):
            need_refinement = True

        return need_refinement

    def refine(self):
        gd = self.gd
        phase = self.phase
        num_nodes = gd.subset_num_nodes['all']
        numseg = gd.num_segments

        # N = phase.options['order']

        P = np.log(self.error/self.tol)/np.log(gd.transcription_order)

        new_order = gd.transcription_order + P
        B = np.ones(numseg)
        if new_order <= self.max_order:
            return new_order, num_nodes, gd.segment_ends
        else:
            new_segment_ends = gd.segment_ends
            new_num_nodes = 0
            for q in range(0, numseg+1):
                B[q] = np.max(int(new_order[q]/self.min_order), 2)
                if B[q] != 1:
                    np.insert(new_segment_ends, np.linspace(new_segment_ends[q], new_segment_ends[q + 1], B[q]), q)
                    new_num_nodes += B[q]*self.min_order
                else:
                    new_num_nodes += gd.transcription_order
            return gd.transcription_order, new_num_nodes, new_segment_ends

    def interpolation_lagrange_matrix(self, old_grid, new_grid):
        L_blocks = []

        for iseg in range(old_grid.num_segments):
            i1, i2 = old_grid.subset_segment_indices['all'][iseg, :]
            indices = old_grid.subset_node_indices['all'][i1:i2]
            nodes_given = old_grid.node_stau[indices]

            i1, i2 = new_grid.subset_segment_indices['all'][iseg, :]
            indices = new_grid.subset_node_indices['all'][i1:i2]
            nodes_eval = new_grid.node_stau[indices][1:]

            L_block, _ = lagrange_matrices(nodes_given, nodes_eval)

            L_blocks.append(L_block)

        L = block_diag(*L_blocks)

        return L

    def integration_matrix(self, grid):
        I_blocks = []

        for iseg in range(grid.num_segments):
            i1, i2 = grid.subset_segment_indices['all'][iseg, :]
            indices = grid.subset_node_indices['all'][i1:i2]
            nodes_given = grid.node_stau[indices]

            i1, i2 = grid.subset_segment_indices['all'][iseg, :]
            indices = grid.subset_node_indices['all'][i1:i2]
            nodes_eval = grid.node_stau[indices][1:]

            _, D_block = lagrange_matrices(nodes_given, nodes_eval)
            I_block = np.linalg.inv(D_block[:, 1:])
            I_blocks.append(I_block)

        I = block_diag(*I_blocks)

        return I

    def eval_ode(self, grid):
        """
        Evaluate the phase ODE on the given grid.
        """
        p = om.Problem(model=om.Group())
        ode_class = self.phase.options['ode_class']
        ode_init_kwargs = self.phase.options['ode_init_kwargs']
        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])
        p.model.add_subsystem('ode', subsys=ode_class(**ode_init_kwargs))

        

