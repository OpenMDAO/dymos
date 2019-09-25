from __future__ import division, print_function, absolute_import

from collections import Iterable, Sequence
from ...transcriptions.grid_data import GridData
from ...transcriptions.common import TimeComp
from ...phase.phase import Phase
from ...utils.lagrange import lagrange_matrices

from scipy.linalg import block_diag

import numpy as np

import openmdao.api as om
from openmdao.core.system import System
import dymos as dm


def interpolation_lagrange_matrix(old_grid, new_grid):
    """
    Evaluate lagrange matrix to interpolate state and control values from the solved grid onto the new grid

    Parameters
    ----------
    old_grid: GridData
        The GridData object representing the grid on which the problem has been solved
    new_grid: GridData
        The GridData object representing the new, higher-order grid

    Returns
    -------
    L: np.ndarray
        The lagrange interpolation matrix

    """
    L_blocks = []

    for iseg in range(old_grid.num_segments):
        i1, i2 = old_grid.subset_segment_indices['all'][iseg, :]
        indices = old_grid.subset_node_indices['all'][i1:i2]
        nodes_given = old_grid.node_stau[indices]

        i1, i2 = new_grid.subset_segment_indices['all'][iseg, :]
        indices = new_grid.subset_node_indices['all'][i1:i2]
        nodes_eval = new_grid.node_stau[indices]

        L_block, _ = lagrange_matrices(nodes_given, nodes_eval)

        L_blocks.append(L_block)

    L = block_diag(*L_blocks)

    return L


def integration_matrix(grid):
    """
    Evaluate the Integration matrix of the given grid.

    Parameters
    ----------
    grid: GridData
        The GridData object representing the grid on which the integration matrix is to be evaluated

    Returns
    -------
    I: np.ndarray
        The integration matrix used to propagate initial states over segments

    """
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


def split_segments(old_seg_ends, B):
    """
    Funcion to compute the new segment ends for the refined grid by splitting the necessary segments

    Parameters
    ----------
    old_seg_ends: np.array
        segment ends of grid on which the problem was solved

    B: np.array of ints
        Number of segments to be split into

    Returns
    -------
    new_segment_ends: np.array
        Segment ends of refined grid

    """
    new_segment_ends = []
    for q in range(0, B.size):
        new_ends = list(np.linspace(old_seg_ends[q], old_seg_ends[q+1], B[q]+1))
        new_segment_ends.extend(new_ends[:-1])
    new_segment_ends.extend([1])
    new_segment_ends = np.asarray(new_segment_ends)
    return new_segment_ends


class PHAdaptive:
    """
    Grid refinement object for the p-then-h grid refinement algorithm

    The error on a solved phase is evaluated. If error exceeds chosen tolerance, the grid is refined following the
    p-then-h refinement algorithm.
    Patterson, M. A., Hager, W. W., and Rao. A. V., “A ph Mesh Refinement Method for Optimal Control”,
    Optimal Control Applications and Methods, Vol. 36, No. 4, July - August 2015, pp. 398 - 421. DOI: 10.1002/oca2114

    """

    def __init__(self, phase, iteration_limit=10, tol=1e-6, min_order=3, max_order=7, plot=False):
        """
        Initialize and compute attributes

        Parameters
        ----------
        phase: Phase
            The Phase object representing the solved phase

        iteration_limit: Maximum number of iterations allowed

        tol: float
            The tolerance on the error for determining whether refinement

        min_order: int
            The minimum allowed order for a refined segment

        max_order: int
            The maximum allowed order for a refined segment

        plot: bool
            Allows for plotting the integrated and interpolated solutions

        """
        self.phase = phase
        self.iteration_limit = iteration_limit
        self.tol = tol
        self.min_order = min_order
        self.max_order = max_order
        # self.gd = phase.options['transcription'].grid_data
        self.error = []
        self.plot = plot

    def check_error(self):
        """
        Compute the error in every solved segment

        Returns
        -------
        need_refinement: np.array bool
            Indicator for which segments of the given phase require grid refinement

        """
        gd = self.phase.options['transcription'].grid_data
        phase = self.phase
        num_nodes = gd.subset_num_nodes['all']
        numseg = gd.num_segments

        outputs = phase.list_outputs(units=False, out_stream=None)

        out_values_dict = {k: v['value'] for k, v in outputs}

        prom_to_abs_map = phase._var_allprocs_prom2abs_list['output']

        num_scalar_states = 0
        for state_name, options in phase.state_options.items():
            shape = options['shape']
            size = np.prod(shape)
            num_scalar_states += size

        x = np.zeros([num_nodes, num_scalar_states])
        f = np.zeros([num_nodes, num_scalar_states])
        c = 0

        # Obtain the solution on the current grid
        for state_name, options in phase.state_options.items():
            prom_name = f'timeseries.states:{state_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            rate_source_prom_name = 'rhs_all.' + options['rate_source']
            rate_abs_name = prom_to_abs_map[rate_source_prom_name][0]
            x[:, c] = out_values_dict[abs_name].ravel()
            f[:, c] = out_values_dict[rate_abs_name].ravel()
            c += 1

        # Obtain the solution on the new grid
        # interpolate x at t_hat
        new_order = gd.transcription_order + 1
        new_grid = GridData(numseg, gd.transcription, new_order, gd.segment_ends, gd.compressed)
        nodes_per_seg_new = new_grid.subset_num_nodes_per_segment['all']

        L = interpolation_lagrange_matrix(gd, new_grid)
        I = integration_matrix(new_grid)

        # Call the ODE at all nodes of the new grid
        x_hat, x_prime = self.eval_ode(new_grid, L, I)
        E = {}
        e = {}
        err_over_states = {}

        for state_name, options in self.phase.state_options.items():
            E[state_name] = np.absolute(x_prime[state_name] - x_hat[state_name])
            for k in range(0, numseg):
                print(k)
                e[state_name] = E[state_name]/(1 + np.max(x_hat[state_name][k*nodes_per_seg_new[k]:(k+1)*nodes_per_seg_new[k]]))
            err_over_states[state_name] = np.zeros(numseg)

        for state_name, options in self.phase.state_options.items():
            for k in range(0, numseg):
                err_over_states[state_name][k] = np.max(e[state_name][k*nodes_per_seg_new[k]:(k+1)*nodes_per_seg_new[k]])

        self.error = np.zeros(numseg)
        need_refinement = np.zeros(numseg, dtype=bool)

        for state_name, options in self.phase.state_options.items():
            for k in range(0, numseg):
                if err_over_states[state_name][k] > self.error[k]:
                    self.error[k] = err_over_states[state_name][k]
                    if self.error[k] > self.tol:
                        need_refinement[k] = True

        return need_refinement

    def refine(self, need_refinement):
        """
        Compute the order, number of nodes, and segment ends required for the new grid

        Returns
        -------

        new_order: int
            Computed new order of the segments

        new_segment_ends: np.array
            New segment ends computed from splitting existing segments

        new_num_nodes: int
            Number of nodes in the refined grid

        """
        gd = self.phase.options['transcription'].grid_data
        numseg = gd.num_segments

        refine_seg_idxs = np.where(need_refinement)

        P = np.zeros(numseg)
        P[refine_seg_idxs] = np.log(self.error[refine_seg_idxs]/self.tol)/np.log(gd.transcription_order[refine_seg_idxs])
        P = np.around(P).astype(int)

        new_order = gd.transcription_order + P
        B = np.ones(numseg, dtype=int)

        raise_order_idxs = np.where(gd.transcription_order + P < self.max_order)
        split_seg_idxs = np.where(gd.transcription_order + P > self.max_order)

        new_order[raise_order_idxs] = gd.transcription_order[raise_order_idxs] + P[raise_order_idxs]
        new_order[split_seg_idxs] = self.min_order

        B[split_seg_idxs] = np.around((gd.transcription_order[split_seg_idxs] + P[split_seg_idxs])/self.min_order).astype(int)

        new_order = np.repeat(new_order, repeats=B)
        new_num_segments = int(np.sum(B))
        new_segment_ends = split_segments(gd.segment_ends, B)

        return new_order, new_num_segments, new_segment_ends

    def eval_ode(self, grid, L, I):
        """
        Evaluate the phase ODE on the given grid.

        Parameters
        ----------
        grid : GridData
            The GridData object representing the grid at which the ODE is to be evaluated.
        L : np.ndarray
            The interpolation matrix used to obtain interpolated values for the states and controls
            on the given grid, using the existing values on the current grid.
        I : np.ndarray
            The integration matrix used to propagate the initial states of segments across the given grid

        Returns
        -------
        x_hat : dict
            Interpolated state values at all nodes of the given grid.
        x_prime : dict
            Evaluted state values at all nodes of the given grid from use of Integration matrix.

        """
        time_units = self.phase.time_options['units']
        p = om.Problem(model=om.Group())
        ode_class = self.phase.options['ode_class']
        ode_init_kwargs = self.phase.options['ode_init_kwargs']
        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        time_comp = TimeComp(num_nodes=grid.num_nodes, node_ptau=grid.node_ptau,
                             node_dptau_dstau=grid.node_dptau_dstau, units=time_units)

        p.model.add_subsystem('time', time_comp, promotes_outputs=['*'], promotes_inputs=['*'])

        p.model.add_subsystem('ode', subsys=ode_class(num_nodes=grid.num_nodes, **ode_init_kwargs))

        u = {}
        x = {}
        x_hat = {}
        u_hat = {}
        f_hat = {}
        x_prime = {}

        outputs = self.phase.list_outputs(units=False, out_stream=None)
        out_values_dict = {k: v['value'] for k, v in outputs}
        prom_to_abs_map = self.phase._var_allprocs_prom2abs_list['output']

        if self.phase.time_options['targets']:
            self.phase.connect('time',
                          ['rhs_all.{0}'.format(t) for t in self.phase.time_options['targets']],
                          src_indices=self.grid_data.subset_node_indices['all'])

        if self.phase.time_options['time_phase_targets']:
            self.phase.connect('time_phase',
                          ['rhs_all.{0}'.format(t) for t in self.phase.time_options['time_phase_targets']],
                          src_indices=self.grid_data.subset_node_indices['all'])

        if self.phase.time_options['t_initial_targets']:
            tgts = self.phase.time_options['t_initial_targets']
            self.phase.connect('t_initial',
                          ['rhs_all.{0}'.format(t) for t in tgts])

        if self.phase.time_options['t_duration_targets']:
            tgts = self.phase.time_options['t_duration_targets']
            self.phase.connect('t_duration',
                          ['rhs_all.{0}'.format(t) for t in tgts])

        for state_name, options in self.phase.state_options.items():
            prom_name = f'timeseries.states:{state_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            x[state_name] = out_values_dict[abs_name]
            x_hat[state_name] = np.dot(L, x[state_name])
            ivc.add_output(f'states:{state_name}', val=x_hat[state_name], units=options['units'])
            p.model.connect(f'states:{state_name}', [f'ode.{tgt}' for tgt in options['targets']])

        for control_name, options in self.phase.control_options.items():
            prom_name = f'timeseries.controls:{control_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            u[control_name] = out_values_dict[abs_name]
            u_hat[control_name] = np.dot(L, u[control_name])
            ivc.add_output(f'controls:{control_name}', val=u_hat[control_name], units=options['units'])
            p.model.connect(f'controls:{control_name}', [f'ode.{tgt}' for tgt in options['targets']])

        for dp_name, options in self.phase.design_parameter_options.items():
            prom_name = f'design_parameters:{dp_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            dp_val = out_values_dict[abs_name]
            ivc.add_output(f'design_parameters:{dp_name}', val=dp_val, units=options['units'])
            p.model.connect(f'design_parameters:{dp_name}',
                            [f'ode.{tgt}' for tgt in options['targets']])

        p.setup()

        ti_prom_name = f't_initial'
        ti_abs_name = prom_to_abs_map[ti_prom_name][0]
        t_initial = out_values_dict[ti_abs_name]

        td_prom_name = f't_duration'
        td_abs_name = prom_to_abs_map[td_prom_name][0]
        t_duration = out_values_dict[td_abs_name]

        p.set_val('t_initial', t_initial)
        p.set_val('t_duration', t_duration)

        p.run_model()

        left_end_idxs = grid.subset_node_indices['segment_ends'][0::2]
        all_idxs = grid.subset_node_indices['all']
        not_left_end_idxs = np.array(sorted(list(set(all_idxs).difference(set(left_end_idxs)))))

        oodt_dstau = np.atleast_2d(p.get_val('dt_dstau')[not_left_end_idxs]).T

        for state_name, options in self.phase.state_options.items():
            rate_source = options['rate_source']
            f_hat[state_name] = np.atleast_2d(p.get_val(f'ode.{rate_source}'))
            if options['shape'] == (1,):
                f_hat[state_name] = f_hat[state_name].T
            x_prime[state_name] = np.zeros((grid.num_nodes,) + options['shape'])
            x_prime[state_name][left_end_idxs, ...] = x_hat[state_name][left_end_idxs, ...]
            nnps = np.array(grid.subset_num_nodes_per_segment['all']) - 1
            left_end_idxs_repeated = np.repeat(left_end_idxs, nnps)

            x_prime[state_name][not_left_end_idxs, ...] = \
                x_hat[state_name][left_end_idxs_repeated, ...] \
                + oodt_dstau * np.dot(I, f_hat[state_name][not_left_end_idxs, ...])

        if self.plot:
            import matplotlib.pyplot as plt
            ptau_old = self.gd.node_ptau
            ptau_new = grid.node_ptau
            plt.plot(ptau_old, x['x'], 'ro', label='x')
            plt.plot(ptau_new, x_hat['x'], 'bx', label='x hat')
            plt.plot(ptau_new, x_prime['x'], 'k.', label='x prime')
            plt.legend()
            plt.show()

        return x_hat, x_prime
