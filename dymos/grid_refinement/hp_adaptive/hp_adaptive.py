from ...transcriptions.grid_data import GridData
from ...transcriptions.common import TimeComp
from ...phase.phase import Phase
from ...utils.lagrange import lagrange_matrices
from ...utils.lgr import lgr
from ...utils.lgl import lgl
from ...utils.interpolate import LagrangeBarycentricInterpolant

from scipy.linalg import block_diag

import numpy as np

import openmdao.api as om
import dymos as dm


def interpolation_lagrange_matrices(old_grid, new_grid):
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

    D: np.ndarray
        The lagrange differentiation matrix

    """
    L_blocks = []
    D_blocks = []

    for iseg in range(old_grid.num_segments):
        i1, i2 = old_grid.subset_segment_indices['all'][iseg, :]
        indices = old_grid.subset_node_indices['all'][i1:i2]
        nodes_given = old_grid.node_stau[indices]

        i1, i2 = new_grid.subset_segment_indices['all'][iseg, :]
        indices = new_grid.subset_node_indices['all'][i1:i2]
        nodes_eval = new_grid.node_stau[indices]

        L_block, D_block = lagrange_matrices(nodes_given, nodes_eval)

        L_blocks.append(L_block)
        D_blocks.append(D_block)

    L = block_diag(*L_blocks)
    D = block_diag(*D_blocks)

    return L, D


def integration_matrix(grid):
    """
    Evaluate the Integration matrix of the given grid.

    Parameters
    ----------
    grid: GridData
        The GridData object representing the grid on which the integration matrix is to be evaluated

    Returns
    -------
    int_matrix: np.ndarray
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

    int_matrix = block_diag(*I_blocks)

    return int_matrix


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
        new_ends = list(np.linspace(old_seg_ends[q], old_seg_ends[q + 1], B[q] + 1))
        new_segment_ends.extend(new_ends[:-1])
    new_segment_ends.extend([1])
    new_segment_ends = np.asarray(new_segment_ends)
    return new_segment_ends


def merge_segments(old_seg_ends, seg_merge):
    """
        Funcion to compute the new segment ends for the refined grid by merging the unnecessary segments

        Parameters
        ----------
        old_seg_ends: np.array
            segment ends of grid on which the problem was solved

        seg_merge: np.array of booleans
            Wether a segment is to be merged or not

        Returns
        -------
        new_seg_ends: np.array
            Segment ends of refined grid

        """
    new_seg_ends = [old_seg_ends[0]]
    for q in range(1, seg_merge.size):
        if seg_merge[q]:
            continue
        new_seg_ends.append(old_seg_ends[q])
    new_seg_ends.append(1)
    new_seg_ends = np.asarray(new_seg_ends)
    return new_seg_ends


class HPAdaptive:
    """
    Grid refinement object for the hp grid refinement algorithm

    The error on a solved phase is evaluated. If error exceeds chosen tolerance, the grid is refined following the
    p-then-h refinement algorithm.
    Patterson, M. A., Hager, W. W., and Rao. A. V., “Adaptive mesh refinement method for optimal control using
    non-smoothness detection and mesh size reduction”, Journal of the Franklin Institute 352 (2015) 4081–4106

    """

    def __init__(self, phases):
        """
        Initialize and compute attributes

        Parameters
        ----------
        phases: Phase
            The Phase object representing the solved phase

        """
        self.phases = phases
        self.error = {}
        self.iteration_number = 0
        self.previous_error = {}
        self.gd = {}
        self.x_dd = {}

    def check_error(self):
        """
        Compute the error in every solved segment

        Returns
        -------
        need_refinement: dict
            Indicator for which segments of the given phase require grid refinement

        """
        refine_results = {}

        if self.iteration_number > 0:
            self.previous_error = self.error

        for phase_path, phase in self.phases.items():
            refine_results[phase_path] = {}

            # Save the original grid to the refine results
            tx = phase.options['transcription']
            gd = tx.grid_data
            num_nodes = gd.subset_num_nodes['all']
            numseg = gd.num_segments

            refine_results[phase_path]['num_segments'] = numseg
            refine_results[phase_path]['order'] = gd.transcription_order
            refine_results[phase_path]['segment_ends'] = gd.segment_ends
            refine_results[phase_path]['need_refinement'] = np.zeros(numseg, dtype=bool)
            refine_results[phase_path]['error'] = np.zeros(numseg, dtype=float)

            if isinstance(tx, dm.RungeKutta):
                continue

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
                rate_source_prom_name = f"timeseries.state_rates:{state_name}"
                rate_abs_name = prom_to_abs_map[rate_source_prom_name][0]
                x[:, c] = out_values_dict[abs_name].ravel()
                f[:, c] = out_values_dict[rate_abs_name].ravel()
                c += 1

            # Obtain the solution on the new grid
            # interpolate x at t_hat
            new_order = gd.transcription_order + 1
            # Gauss-Lobatto does not allow even orders so increase order by 2 instead
            if gd.transcription == 'gauss-lobatto':
                new_order += 1
            new_grid = GridData(numseg, gd.transcription, new_order, gd.segment_ends, gd.compressed)
            left_end_idxs = new_grid.subset_node_indices['segment_ends'][0::2]
            left_end_idxs = np.append(left_end_idxs, new_grid.subset_num_nodes['all'] - 1)

            L, _ = interpolation_lagrange_matrices(gd, new_grid)
            int_matrix = integration_matrix(new_grid)

            # Call the ODE at all nodes of the new grid
            x_hat, x_prime, _, _, _ = self.eval_ode(phase, new_grid, L, int_matrix)
            E = {}
            e = {}
            err_over_states = {}
            for state_name, options in phase.state_options.items():
                E[state_name] = np.absolute(x_prime[state_name] - x_hat[state_name])
                e[state_name] = np.zeros(E[state_name].shape)
                for k in range(0, numseg):
                    e[state_name][left_end_idxs[k]:left_end_idxs[k + 1]] = \
                        E[state_name][left_end_idxs[k]:left_end_idxs[k + 1]] / \
                        (1 + np.max(x_hat[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))

                err_over_states[state_name] = np.zeros(numseg)

            for state_name, options in phase.state_options.items():
                for k in range(0, numseg):
                    err_over_states[state_name][k] = np.max(e[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])

            self.error[phase_path] = np.zeros(numseg)
            refine_results[phase_path]['error'] = np.zeros(numseg)
            refine_results[phase_path]['need_refinement'] = np.zeros(numseg, dtype=bool)

            # Assess the errors in each state
            for state_name, options in phase.state_options.items():
                for k in range(0, numseg):
                    if err_over_states[state_name][k] > self.error[phase_path][k]:
                        self.error[phase_path][k] = err_over_states[state_name][k]
                        refine_results[phase_path]['error'][k] = err_over_states[state_name][k]
                        if self.error[phase_path][k] > phase.refine_options['tolerance']:
                            refine_results[phase_path]['need_refinement'][k] = True

        return refine_results

    def refine_first_iter(self, refine_results):
        """
            Compute the order, number of nodes, and segment ends required for the new grid
            and assigns them to the transcription of each phase. Method of refinement is
            different for the first iteration and is done separately

            Parameters
            ----------
            refine_results : dict
                A dictionary where each key is the path to a phase in the problem, and the
                associated value are various properties of that phase needed by the refinement
                algorithm.  refine_results is returned by check_error.  This method modifies it
                in place, adding the new_num_segments, new_order, and new_segment_ends.

            Returns
            -------
            refined : dict
                A dictionary of phase paths : phases which were refined.

        """

        for phase_path, phase_refinement_results in refine_results.items():
            phase = self.phases[phase_path]
            tx = phase.options['transcription']
            gd = tx.grid_data
            self.gd[phase_path] = gd
            self.x_dd[phase_path] = {}

            num_scalar_states = 0
            for state_name, options in phase.state_options.items():
                shape = options['shape']
                size = np.prod(shape)
                num_scalar_states += size

            seg_order = gd.transcription_order
            seg_ends = gd.segment_ends

            need_refine = phase_refinement_results['need_refinement']
            if not phase.refine_options['refine'] or not np.any(need_refine):
                refine_results[phase_path]['new_order'] = seg_order
                refine_results[phase_path]['new_num_segments'] = gd.num_segments
                refine_results[phase_path]['new_segment_ends'] = seg_ends
                continue

            left_end_idxs = gd.subset_node_indices['segment_ends'][0::2]
            left_end_idxs = np.append(left_end_idxs, gd.subset_num_nodes['all'] - 1)

            # obtain state and state rate histories from timeseries output
            L, D = interpolation_lagrange_matrices(gd, gd)
            int_matrix = integration_matrix(gd)
            x, _, x_d, _, _ = self.eval_ode(phase, gd, L, int_matrix)

            # create and store second derivative information using differentiation matrix
            for state_name, options in phase.state_options.items():
                self.x_dd[phase_path][state_name] = D @ x_d[state_name]

            # Refinement is needed
            gd = phase.options['transcription'].grid_data
            numseg = gd.num_segments

            merge_seg = np.zeros(numseg, dtype=bool)

            # In the first iteration no segments are split. All segments not meeting error requirements have their order
            # increased
            inc_seg_order_idxs = np.where(need_refine)
            P = np.zeros(numseg)
            if gd.transcription == 'radau-ps':
                P[inc_seg_order_idxs] = 3
            elif gd.transcription == 'gauss-lobatto':
                P[inc_seg_order_idxs] = 2  # GL transcription does not allow even segment orders
            new_order = (seg_order + P).astype(int)

            h = 0.5 * (seg_ends[1:] - seg_ends[:-1])

            # Segments which should be checked for combining
            # Only segments where adjacent segments are also below error tolerance may be combined
            # Segments must be same order
            check_comb_indx = np.where(np.logical_and(np.logical_and(np.invert(need_refine[:-1]),
                                                                     np.invert(need_refine[1:])),
                                                      new_order[:-1] == new_order[1:]))

            # segments under error tolerance but may not be combined have their order reduced
            reduce_order_indx = np.setdiff1d(np.where(np.invert(need_refine)), check_comb_indx)

            # reduce segment order where error is much below the tolerance
            for k in np.nditer(reduce_order_indx):
                print(k)
                new_order[k] = seg_order[k]
                a = np.zeros((seg_order[k] + 1, seg_order[k] + 1))
                s, _ = lgr(seg_order[k], include_endpoint=True)
                if gd.transcription == 'gauss-lobatto':
                    s, _ = lgl(seg_order[k], include_endpoint=True)
                for j in range(0, seg_order[k] + 1):
                    roots = s[s != s[j]]
                    print(s)
                    print(s[s!=s[j]])
                    Q = np.poly(roots)
                    a[:, j] = Q / np.polyval(Q, s[j])

                for state_name, options in phase.state_options.items():
                    beta = 1 + np.max(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])
                    b = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]

                    for i in range(seg_order[k] - 1, 1, -1):
                        if b[i] / beta > phase.refine_options['tolerance'] and i - 1 < new_order[k]:
                            new_order[k] = i - 1

            # combine unnecessary segments
            for k in np.nditer(check_comb_indx):
                a = np.zeros((new_order[k] + 1, new_order[k] + 1))
                h_ = np.maximum(h[k], h[k + 1])
                s, _ = lgr(new_order[k].astype(int), include_endpoint=True)
                if gd.transcription == 'gauss-lobatto':
                    s, _ = lgl(new_order[k], include_endpoint=True)
                for j in range(0, new_order[k] + 1):
                    roots = s[s != s[j]]
                    Q = np.poly(roots)
                    a[:, j] = Q / np.polyval(Q, s[j])

                merge_seg[k + 1] = True

                for state_name, options in phase.state_options.items():
                    beta = 1 + np.max(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])
                    c = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]
                    b = np.multiply(c.ravel(), np.array([(h_ / h[k]) ** l for l in range(new_order[k] + 1)]))
                    b_hat = np.multiply(c.ravel(), np.array([(h_ / h[k + 1]) ** l for l in range(new_order[k] + 1)]))
                    err_val = np.dot(np.absolute(b - b_hat).ravel(),
                                     np.array([2 ** l for l in range(new_order[k] + 1)])) / beta

                    if err_val > phase.refine_options['tolerance'] and merge_seg[k + 1]:
                        merge_seg[k + 1] = False

            new_segment_ends = merge_segments(gd.segment_ends, merge_seg)
            new_num_segments = new_segment_ends.shape[0] - 1
            new_order = np.delete(new_order, np.where(merge_seg), axis=None)

            refine_results[phase_path]['new_order'] = new_order
            refine_results[phase_path]['new_num_segments'] = new_num_segments
            refine_results[phase_path]['new_segment_ends'] = new_segment_ends

            tx.options['order'] = new_order
            tx.options['num_segments'] = new_num_segments
            tx.options['segment_ends'] = new_segment_ends
            tx.init_grid()

    def refine(self, refine_results):
        """
                    Compute the order, number of nodes, and segment ends required for the new grid
                    and assigns them to the transcription of each phase. Method of refinement is
                    different for the first iteration and is done separately

                    Parameters
                    ----------
                    refine_results : dict
                        A dictionary where each key is the path to a phase in the problem, and the
                        associated value are various properties of that phase needed by the refinement
                        algorithm.  refine_results is returned by check_error.  This method modifies it
                        in place, adding the new_num_segments, new_order, and new_segment_ends.

                    Returns
                    -------
                    refined : dict
                        A dictionary of phase paths : phases which were refined.

        """
        x_dd = {}
        for phase_path, phase_refinement_results in refine_results.items():
            phase = self.phases[phase_path]
            tx = phase.options['transcription']
            gd = tx.grid_data

            num_scalar_states = 0
            for state_name, options in phase.state_options.items():
                shape = options['shape']
                size = np.prod(shape)
                num_scalar_states += size

            seg_order = gd.transcription_order
            seg_ends = gd.segment_ends
            numseg = gd.num_segments

            need_refine = phase_refinement_results['need_refinement']
            if not phase.refine_options['refine'] or not np.any(need_refine):
                refine_results[phase_path]['new_order'] = seg_order
                refine_results[phase_path]['new_num_segments'] = gd.num_segments
                refine_results[phase_path]['new_segment_ends'] = seg_ends
                continue

            left_end_idxs = gd.subset_node_indices['segment_ends'][0::2]
            left_end_idxs = np.append(left_end_idxs, gd.subset_num_nodes['all'])
            refine_seg_idxs = np.where(need_refine)[0]

            # compute curvature
            L, D = interpolation_lagrange_matrices(gd, gd)
            int_matrix = integration_matrix(gd)
            x, _, x_d, t_initial, t_duration = self.eval_ode(phase, gd, L, int_matrix)
            x_dd[phase_path] = {}
            P = {}
            P_hat = {}
            R = np.zeros(numseg)

            for state_name, options in phase.state_options.items():
                interp = LagrangeBarycentricInterpolant(self.gd[phase_path].node_ptau, options['shape'])
                interp.setup(x0=t_initial, xf=t_initial + t_duration, f_j=self.x_dd[phase_path][state_name])
                x_dd[phase_path][state_name] = D @ x_d[state_name]
                P[state_name] = np.zeros(numseg)
                P_hat[state_name] = np.zeros(numseg)
                for k in np.nditer(refine_seg_idxs):
                    P[state_name][k] = np.max(
                        np.absolute(x_dd[phase_path][state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))
                    xdd_max_time = gd.node_ptau[np.argmax(
                        np.absolute(x_dd[phase_path][state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))]
                    P_hat[state_name][k] = interp.eval(xdd_max_time)
                    P_hat[state_name][k] = np.max(
                        np.absolute(self.x_dd[phase_path][state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))
                    if P[state_name][k] / P_hat[state_name][k] > R[k]:
                        R[k] = P[state_name][k] / P_hat[state_name][k]

            non_smooth_idxs = np.where(R > phase.refine_options['smoothness_factor'])[0]
            smooth_need_refine_idxs = np.setdiff1d(refine_seg_idxs, non_smooth_idxs)

            P = np.ones(numseg)

            P[smooth_need_refine_idxs] = (
                    self.error[phase_path][smooth_need_refine_idxs] / phase.refine_options['tolerance'])

            new_order = np.ceil(gd.transcription_order * P).astype(int)
            if gd.transcription == 'gauss-lobatto':
                odd_idxs = np.where(new_order % 2 != 0)
                new_order[odd_idxs] += 1

            H = np.ones(numseg, dtype=int)
            q = np.zeros(numseg)
            h = 0.5 * (seg_ends[1:] - seg_ends[:-1])
            h_prev = 0.5 * (self.gd[phase_path].segment_ends[1:] - self.gd[phase_path].segment_ends[:-1])

            split_seg_idxs = np.concatenate([np.where(new_order > phase.refine_options['max_order'])[0],
                                             non_smooth_idxs])

            check_comb_indx = np.where(np.logical_and(np.logical_and(np.invert(need_refine[:-1]),
                                                                     np.invert(need_refine[1:])),
                                                      new_order[:-1] == new_order[1:]))[0]

            reduce_order_indx = np.setdiff1d(np.where(np.invert(need_refine)), check_comb_indx)

            new_order[split_seg_idxs] = seg_order[split_seg_idxs]

            q[split_seg_idxs] = np.log((self.error[phase_path][split_seg_idxs] /
                                        self.previous_error[phase_path][split_seg_idxs]) /
                                       (seg_order[split_seg_idxs] / self.gd[phase_path].transcription_order[
                                           split_seg_idxs]) ** 2.5
                                       ) / np.log((h[split_seg_idxs] / h_prev[split_seg_idxs])
                                                  / (seg_order[split_seg_idxs] /
                                                     self.gd[phase_path].transcription_order[split_seg_idxs]))

            H[split_seg_idxs] = np.minimum(
                np.ceil((self.error[phase_path][split_seg_idxs] / phase.refine_options['tolerance']
                         ) ** (1 / q[split_seg_idxs])), np.ceil(np.log(self.error[phase_path][split_seg_idxs] /
                                                                       phase.refine_options['tolerance']) / np.log(
                    seg_order[split_seg_idxs])))

            # reduce segment order where error is much below the tolerance
            if len(list(reduce_order_indx)) != 0:
                for k in np.nditer(reduce_order_indx):
                    if seg_order[k] == 1:
                        continue
                    new_order[k] = seg_order[k]
                    a = np.zeros((seg_order[k] + 1, seg_order[k] + 1))
                    s, _ = lgr(seg_order[k], include_endpoint=True)
                    if gd.transcription == 'gauss-lobatto':
                        s, _ = lgl(seg_order[k], include_endpoint=True)
                    for j in range(0, seg_order[k] + 1):
                        roots = s[s != s[j]]
                        Q = np.poly(roots)
                        a[:, j] = Q / np.polyval(Q, s[j])
                    for state_name, options in phase.state_options.items():
                        beta = 1 + np.max(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])
                        b = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]

                        for i in range(seg_order[k] - 1, 1, -1):
                            if b[i] / beta > phase.refine_options['tolerance'] and i - 1 < new_order[k]:
                                new_order[k] = i - 1

            # combine unnecessary segments
            merge_seg = np.zeros(numseg, dtype=bool)
            if len(list(check_comb_indx)) != 0:
                for k in np.nditer(check_comb_indx):
                    a = np.zeros((new_order[k] + 1, new_order[k] + 1))
                    h_ = np.maximum(h[k], h[k + 1])
                    s, _ = lgr(new_order[k].astype(int), include_endpoint=True)
                    if gd.transcription == 'gauss-lobatto':
                        s, _ = lgl(new_order[k], include_endpoint=True)
                    for j in range(0, new_order[k] + 1):
                        roots = s[s != s[j]]
                        Q = np.poly(roots)
                        a[:, j] = Q / np.polyval(Q, s[j])

                    merge_seg[k + 1] = True

                    for state_name, options in phase.state_options.items():
                        beta = 1 + np.max(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])
                        c = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]
                        b = np.multiply(c.ravel(), np.array([(h_ / h[k]) ** l for l in range(new_order[k] + 1)]))
                        b_hat = np.multiply(c.ravel(),
                                            np.array([(h_ / h[k + 1]) ** l for l in range(new_order[k] + 1)]))
                        err_val = np.dot(np.absolute(b - b_hat).ravel(),
                                         np.array([2 ** l for l in range(new_order[k] + 1)])) / beta

                        if err_val > phase.refine_options['tolerance'] and merge_seg[k + 1]:
                            merge_seg[k + 1] = False

            H[np.where(merge_seg)] = 0

            new_order = np.repeat(new_order, repeats=H)
            new_num_segments = int(np.sum(H))
            new_segment_ends = split_segments(gd.segment_ends, H)

            refine_results[phase_path]['new_order'] = new_order
            refine_results[phase_path]['new_num_segments'] = new_num_segments
            refine_results[phase_path]['new_segment_ends'] = new_segment_ends

            tx.options['order'] = new_order
            tx.options['num_segments'] = new_num_segments
            tx.options['segment_ends'] = new_segment_ends
            tx.init_grid()
            self.x_dd[phase_path] = x_dd[phase_path]
            self.gd[phase_path] = gd

    def eval_ode(self, phase, grid, L, I):
        """
        Evaluate the phase ODE on the given grid.

        Parameters
        ----------
        phase : Phase
            The phase object whose ODE is to be evaluated at the given grid.
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
        time_units = phase.time_options['units']
        grid_data = phase.options['transcription'].grid_data
        p = om.Problem(model=om.Group())
        ode_class = phase.options['ode_class']
        ode_init_kwargs = phase.options['ode_init_kwargs']
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

        outputs = phase.list_outputs(units=False, out_stream=None)
        values_dict = {k: v['value'] for k, v in outputs}
        prom_to_abs_map = phase._var_allprocs_prom2abs_list['output']

        inputs = phase.list_inputs(units=False, out_stream=None)
        prom_to_abs_map.update(phase._var_allprocs_prom2abs_list['input'])
        values_dict.update({k: v['value'] for k, v in inputs})

        if phase.time_options['targets']:
            p.model.connect('time',
                            [f'ode.{t}' for t in phase.time_options['targets']])

        if phase.time_options['time_phase_targets']:
            p.model.connect('time_phase',
                            [f'ode.{t}' for t in phase.time_options['time_phase_targets']],
                            src_indices=grid_data.subset_node_indices['all'])

        if phase.time_options['t_initial_targets']:
            tgts = phase.time_options['t_initial_targets']
            p.model.connect('t_initial',
                            [f'ode.{t}' for t in tgts])

        if phase.time_options['t_duration_targets']:
            tgts = phase.time_options['t_duration_targets']
            p.model.connect('t_duration',
                            [f'ode.{t}' for t in tgts])

        for state_name, options in phase.state_options.items():
            prom_name = f'timeseries.states:{state_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            x[state_name] = values_dict[abs_name]
            x_hat[state_name] = np.dot(L, x[state_name])
            ivc.add_output(f'states:{state_name}', val=x_hat[state_name], units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'states:{state_name}', [f'ode.{tgt}' for tgt in options['targets']])

        for control_name, options in phase.control_options.items():
            prom_name = f'timeseries.controls:{control_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            u[control_name] = values_dict[abs_name]
            u_hat[control_name] = np.dot(L, u[control_name])
            ivc.add_output(f'controls:{control_name}', val=u_hat[control_name], units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'controls:{control_name}', [f'ode.{tgt}' for tgt in options['targets']])

        for dp_name, options in phase.design_parameter_options.items():
            prom_name = f'design_parameters:{dp_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            dp_val = values_dict[abs_name][0, ...]
            ivc.add_output(f'design_parameters:{dp_name}', val=dp_val, units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'design_parameters:{dp_name}',
                                [f'ode.{tgt}' for tgt in options['targets']],
                                src_indices=np.zeros(grid.num_nodes, dtype=int))

        for dp_name, options in phase.input_parameter_options.items():
            prom_name = f'input_parameters:{dp_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            dp_val = values_dict[abs_name][0, ...]
            ivc.add_output(f'input_parameters:{dp_name}', val=dp_val, units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'input_parameters:{dp_name}',
                                [f'ode.{tgt}' for tgt in options['targets']],
                                src_indices=np.zeros(grid.num_nodes, dtype=int))

        p.setup()

        ti_prom_name = f't_initial'
        ti_abs_name = prom_to_abs_map[ti_prom_name][0]
        t_initial = values_dict[ti_abs_name]

        td_prom_name = f't_duration'
        td_abs_name = prom_to_abs_map[td_prom_name][0]
        t_duration = values_dict[td_abs_name]

        p.set_val('t_initial', t_initial)
        p.set_val('t_duration', t_duration)

        p.run_model()

        left_end_idxs = grid.subset_node_indices['segment_ends'][0::2]
        all_idxs = grid.subset_node_indices['all']
        not_left_end_idxs = np.array(sorted(list(set(all_idxs).difference(set(left_end_idxs)))))

        oodt_dstau = np.atleast_2d(p.get_val('dt_dstau')[not_left_end_idxs]).T

        for state_name, options in phase.state_options.items():
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

        return x_hat, x_prime, f_hat, t_initial, t_duration

    def write_iteration(self, f, iter_number, phases, refine_results):
        """
        Writes a summary of the current grid refinement iteration to the given stream.

        Parameters
        ----------
        f : stream
            The output stream to which the grid refinment should be printed.
        iter_number : int
            The current grid refinement iteration index.
        phases : dict of {phase_path: Phase}
            The phases in the problem being refined.
        refine_results : dict
            A dictionary containing the grid refinement data for each phase, keyed by the phase path
            in the model.
        """
        f.write('\n\n')
        print(50 * '=', file=f)
        str_gr = f'Grid Refinement - Iteration {iter_number}'
        print(f'{str_gr:^50}', file=f)
        print(50 * '-', file=f)
        for phase_path, phase in phases.items():
            refine_data = refine_results[phase_path]
            refine_options = phase.refine_options

            f.write('    Phase: {}\n'.format(phase_path))

            # Print the phase grid-refinement settings
            print('        Refinement Options:', file=f)
            print('            Allow Refinement = {}'.format(refine_options['refine']), file=f)
            print('            Tolerance = {}'.format(refine_options['tolerance']), file=f)
            print('            Min Order = {}'.format(refine_options['min_order']), file=f)
            print('            Max Order = {}'.format(refine_options['max_order']), file=f)

            # Print the original grid specs
            print('        Original Grid:', file=f)
            print('            Number of Segments = {}'.format(refine_data['num_segments']), file=f)

            str_segends = ', '.join(str(round(elem, 4)) for elem in refine_data['segment_ends'])
            print(f'            Segment Ends = [{str_segends}]', file=f)

            str_segorders = ', '.join(str(elem) for elem in refine_data['order'])
            print(f'            Segment Order = [{str_segorders}]', file=f)

            error = refine_data['error']
            str_errors = ', '.join(f'{elem:8.4g}' for elem in error)
            print(f'            Error = [{str_errors}]', file=f)

            # Print the modified grid specs
            print('        New Grid:', file=f)
            print('            Number of Segments = {}'.format(refine_data['new_num_segments']), file=f)

            str_segends = ', '.join(str(round(elem, 4)) for elem in refine_data['new_segment_ends'])
            print(f'            Segment Ends = [{str_segends}]', file=f)

            new_order = refine_data['new_order']
            str_segorders = ', '.join(str(elem) for elem in new_order)

            print(f'            Segment Order = [{str_segorders}]', file=f)

            is_refined = True if np.any(refine_data['need_refinement']) and refine_options['refine'] else False
            print(f'        Refined: {is_refined}', file=f)
            print(file=f)
