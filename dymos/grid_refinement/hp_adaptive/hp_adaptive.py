import copy

import numpy as np

from ...utils.lgr import lgr
from ...utils.lgl import lgl
from ...utils.interpolate import LagrangeBarycentricInterpolant
from ..error_estimation import interpolation_lagrange_matrix, eval_ode_on_grid


def split_segments(old_seg_ends, B):
    """
    Compute the new segment ends for the refined grid by splitting the necessary segments.

    Parameters
    ----------
    old_seg_ends : ndarray
        Segment ends of grid on which the problem was solved.
    B : ndrray of int
        Number of segments to be split into.

    Returns
    -------
    ndarray
        Segment ends of refined grid.
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
    Compute the new segment ends for the refined grid by merging the unnecessary segments.

    Parameters
    ----------
    old_seg_ends : ndarray
        Segment ends of grid on which the problem was solved.
    seg_merge : ndarray of bool
        True for segments that are to be merged.

    Returns
    -------
    ndarray
        Segment ends of refined grid.
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
    Grid refinement object for the hp grid refinement algorithm.

    The error on a solved phase is evaluated. If error exceeds chosen tolerance, the grid is refined following the
    p-then-h refinement algorithm.
    Patterson, M. A., Hager, W. W., and Rao. A. V., “Adaptive mesh refinement method for optimal control using
    non-smoothness detection and mesh size reduction”, Journal of the Franklin Institute 352 (2015) 4081–4106

    Parameters
    ----------
    phases : Phase
        The Phase object representing the solved phase.
    """

    def __init__(self, phases):
        self.phases = phases
        self.error = {}
        self.iteration_number = 0
        self.previous_error = {}
        self.previous_gd = {}
        self.previous_x_dd = {}
        self.parent_seg_map = {}

    def refine_first_iter(self, refine_results):
        """
        Refine the grid during the first iteration.

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
        dict
            A dictionary of phase paths : phases which were refined.
        """

        for phase_path, phase_refinement_results in refine_results.items():
            phase = self.phases[phase_path]
            tx = phase.options['transcription']
            gd = tx.grid_data
            self.previous_gd[phase_path] = gd
            self.previous_x_dd[phase_path] = {}
            self.error[phase_path] = refine_results[phase_path]['max_rel_error']

            # Get information about current grid
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
            left_end_idxs = np.append(left_end_idxs, gd.subset_num_nodes['all'])

            # obtain state and state rate histories from timeseries output
            L, D = interpolation_lagrange_matrix(gd, gd)
            x, _, _, x_d = eval_ode_on_grid(phase=phase, transcription=tx)

            # create and store second derivative information using differentiation matrix
            for state_name, options in phase.state_options.items():
                self.previous_x_dd[phase_path][state_name] = D @ x_d[state_name]

            gd = phase.options['transcription'].grid_data
            numseg = gd.num_segments

            merge_seg = np.zeros(numseg, dtype=bool)

            # In the first iteration no segments are split. All segments not meeting error requirements have their order
            # increased by 3 for Radau transcription and by 2 for GL
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
            check_comb_indx = np.where(np.logical_and(np.logical_and(np.logical_and(np.invert(need_refine[:-1]),
                                                                                    np.invert(need_refine[1:])),
                                                                     new_order[:-1] == new_order[1:]),
                                                      new_order[:-1] == phase.refine_options['min_order']))[0]

            # segments under error tolerance but may not be combined are checked to have their order reduced
            reduce_order_indx = np.setdiff1d(np.where(np.invert(need_refine)), check_comb_indx)

            # reduce segment order where error is much below the tolerance
            # Order reduction is done by creating a power series representation of the state data on the segment
            # Series order is progressively reduced until removal of additional terms would lead to error > tolerance
            if reduce_order_indx.size > 0:
                # compute normalization factor beta
                beta = {}
                for state_name, options in phase.state_options.items():
                    beta[state_name] = 0
                    for k in range(0, numseg):
                        beta_seg = np.max(np.abs(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))
                        if beta_seg > beta[state_name]:
                            beta[state_name] = beta_seg
                    beta[state_name] += 1

                for k in np.nditer(reduce_order_indx):
                    seg_size = {'radau-ps': seg_order[k] + 1, 'gauss-lobatto': seg_order[k]}
                    if seg_order[k] == phase.refine_options['min_order']:
                        continue
                    new_order_state = {}
                    new_order[k] = seg_order[k]
                    a = np.zeros((seg_size[gd.transcription], seg_size[gd.transcription]))
                    s, _ = lgr(seg_order[k], include_endpoint=True)
                    if gd.transcription == 'gauss-lobatto':
                        s, _ = lgl(seg_order[k])
                    for j in range(0, seg_size[gd.transcription]):
                        roots = s[s != s[j]]
                        Q = np.poly(roots)
                        a[:, j] = Q / np.polyval(Q, s[j])

                    for state_name, options in phase.state_options.items():
                        b = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]

                        for i in range(seg_size[gd.transcription] - 1, phase.refine_options['min_order'], -1):
                            if np.abs(b[i]) / beta[state_name] < phase.refine_options['tolerance'] and \
                                    i - 1 < new_order_state[state_name]:
                                new_order_state[state_name] = i - 1
                            else:
                                new_order_state[state_name] = seg_order[k]
                        new_order[k] = max(new_order_state.values())

            # combine unnecessary segments
            # The first of the two segments is extrapolated onto the second segment
            # The extrapolation is checked against the current solution
            # If they match closely enough the segments may be merged
            if check_comb_indx.size > 0:
                for k in np.nditer(check_comb_indx):
                    seg_size = {'radau-ps': seg_order[k] + 1, 'gauss-lobatto': seg_order[k]}
                    if merge_seg[k]:
                        continue
                    a = np.zeros((seg_size[gd.transcription], seg_size[gd.transcription]))
                    h_ = np.maximum(h[k], h[k + 1])
                    s, _ = lgr(new_order[k].astype(int), include_endpoint=True)
                    if gd.transcription == 'gauss-lobatto':
                        s, _ = lgl(new_order[k])
                    for j in range(0, seg_size[gd.transcription]):
                        roots = s[s != s[j]]
                        Q = np.poly(roots)
                        a[:, j] = Q / np.polyval(Q, s[j])

                    merge_seg[k + 1] = True

                    for state_name, options in phase.state_options.items():
                        beta = 1 + np.max(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])
                        c = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]
                        b = np.multiply(c.ravel(), np.array([(h_ / h[k]) ** l for l in
                                                             range(seg_size[gd.transcription])]))
                        b_hat = np.multiply(c.ravel(),
                                            np.array([(h_ / h[k + 1]) ** l for l in range(seg_size[gd.transcription])]))
                        err_val = np.dot(np.absolute(b - b_hat).ravel(),
                                         np.array([2 ** l for l in range(seg_size[gd.transcription])])) / beta

                        if err_val > phase.refine_options['tolerance'] and merge_seg[k + 1]:
                            merge_seg[k + 1] = False

            new_segment_ends = merge_segments(gd.segment_ends, merge_seg)
            new_num_segments = new_segment_ends.shape[0] - 1
            new_order = np.delete(new_order, np.where(merge_seg), axis=None)

            # Create a dictionary with the information on which segment included the time of each segment
            self.parent_seg_map[phase_path] = np.zeros(new_num_segments, dtype=int)
            for i in range(1, new_num_segments):
                for j in range(1, numseg):
                    if new_segment_ends[i] == gd.segment_ends[j]:
                        self.parent_seg_map[phase_path][i] = int(j)
                        break
                    elif gd.segment_ends[j - 1] < new_segment_ends[i] < gd.segment_ends[j]:
                        self.parent_seg_map[phase_path][i] = int(j - 1)
                        break

            if gd.transcription == 'gauss-lobatto':
                new_order[(new_order % 2) == 0] = new_order[(new_order % 2) == 0] + 1

            refine_results[phase_path]['new_order'] = new_order
            refine_results[phase_path]['new_num_segments'] = new_num_segments
            refine_results[phase_path]['new_segment_ends'] = new_segment_ends

            tx.options['order'] = new_order
            tx.options['num_segments'] = new_num_segments
            tx.options['segment_ends'] = new_segment_ends
            tx.init_grid()
            self.previous_error[phase_path] = self.error[phase_path].copy()

    def refine(self, refine_results, iter_number):
        """
        Refine the grid during subsequent iterations.

        Compute the order, number of nodes, and segment ends required for the new grid
        and assigns them to the transcription of each phase. Method of refinement is
        different for the first iteration and is done separately.

        Parameters
        ----------
        refine_results : dict
            A dictionary where each key is the path to a phase in the problem, and the
            associated value are various properties of that phase needed by the refinement
            algorithm.  refine_results is returned by check_error.  This method modifies it
            in place, adding the new_num_segments, new_order, and new_segment_ends.
        iter_number : int
            Current iteration of the grid refinement.

        Returns
        -------
        dict
            A dictionary of phase paths : phases which were refined.
        """
        if iter_number == 0:
            self.refine_first_iter(refine_results)
            return

        x_dd = {}
        for phase_path, phase_refinement_results in refine_results.items():
            phase = self.phases[phase_path]
            self.error[phase_path] = refine_results[phase_path]['max_rel_error']
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

            old_left_end_idxs = self.previous_gd[phase_path].subset_node_indices['segment_ends'][0::2]
            old_left_end_idxs = np.append(old_left_end_idxs, self.previous_gd[phase_path].subset_num_nodes['all'])

            # compute curvature
            L, D = interpolation_lagrange_matrix(gd, gd)
            x, _, _, x_d = eval_ode_on_grid(phase=phase, transcription=tx)
            x_dd[phase_path] = {}
            P = {}
            P_hat = {}
            R = np.zeros(numseg)

            # Compute the maximum magnitude of the second derivative of each state
            # Find the same value at the same time on the previous solution
            # If the ratio of these two values for a given state is highest, it is stored as the curvature
            for state_name, options in phase.state_options.items():
                x_dd[phase_path][state_name] = D @ x_d[state_name]
                P[state_name] = np.zeros(numseg)
                P_hat[state_name] = np.zeros(numseg)
                for k in np.nditer(refine_seg_idxs):
                    interp = LagrangeBarycentricInterpolant(
                        self.previous_gd[phase_path].node_stau[old_left_end_idxs[self.parent_seg_map[phase_path][k]]:
                                                               old_left_end_idxs[self.parent_seg_map[phase_path][k] +
                                                                                 1]],
                        options['shape'])
                    interp.setup(x0=-1, xf=1,
                                 f_j=self.previous_x_dd[phase_path][state_name][
                                     old_left_end_idxs[self.parent_seg_map[phase_path][k]]:old_left_end_idxs[
                                         self.parent_seg_map[phase_path][k] + 1]])
                    P[state_name][k] = np.max(
                        np.fabs(x_dd[phase_path][state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))
                    xdd_max_time = gd.node_stau[left_end_idxs[k] + np.argmax(np.amax(
                        np.fabs(x_dd[phase_path][state_name][left_end_idxs[k]:left_end_idxs[k + 1]]), axis=1), axis=0)]
                    P_hat[state_name][k] = np.amax(np.fabs(interp.eval(xdd_max_time)))
                    if P[state_name][k] / P_hat[state_name][k] > R[k]:
                        R[k] = P[state_name][k] / P_hat[state_name][k]
            non_smooth_idxs = np.where(R > phase.refine_options['smoothness_factor'])[0]
            smooth_need_refine_idxs = np.setdiff1d(refine_seg_idxs, non_smooth_idxs)

            mul_factor = np.ones(numseg)
            h = 0.5 * (seg_ends[1:] - seg_ends[:-1])
            H = np.ones(numseg, dtype=int)
            h_prev = 0.5 * (self.previous_gd[phase_path].segment_ends[1:] - self.previous_gd[phase_path].segment_ends[:-1])

            split_parent_seg_idxs = self.parent_seg_map[phase_path][smooth_need_refine_idxs]

            q_smooth = (np.log(self.error[phase_path][smooth_need_refine_idxs] /
                               self.previous_error[phase_path][split_parent_seg_idxs]) +
                        2.5 * np.log(seg_order[smooth_need_refine_idxs] /
                                     self.previous_gd[phase_path].transcription_order[split_parent_seg_idxs])
                        ) / (np.log((h[smooth_need_refine_idxs] / h_prev[split_parent_seg_idxs])) +
                             np.log(seg_order[smooth_need_refine_idxs] /
                                    self.previous_gd[phase_path].transcription_order[split_parent_seg_idxs]))

            q_smooth[q_smooth < 3] = 3.0
            q_smooth[np.isposinf(q_smooth)] = 3.0
            mul_factor[smooth_need_refine_idxs] = (self.error[phase_path][smooth_need_refine_idxs] /
                                                   phase.refine_options['tolerance']) ** \
                                                  (1 / (q_smooth - 2.5))

            new_order = np.ceil(gd.transcription_order * mul_factor).astype(int)
            if gd.transcription == 'gauss-lobatto':
                odd_idxs = np.where(new_order % 2 != 0)
                new_order[odd_idxs] += 1

            split_seg_idxs = np.concatenate([np.where(new_order > phase.refine_options['max_order'])[0],
                                             non_smooth_idxs])

            check_comb_indx = np.where(np.logical_and(np.logical_and(np.logical_and(np.invert(need_refine[:-1]),
                                                                                    np.invert(need_refine[1:])),
                                                                     new_order[:-1] == new_order[1:]),
                                                      new_order[:-1] == phase.refine_options['min_order']))[0]

            reduce_order_indx = np.setdiff1d(np.where(np.invert(need_refine)), check_comb_indx)

            new_order[split_seg_idxs] = seg_order[split_seg_idxs]
            split_parent_seg_idxs = self.parent_seg_map[phase_path][split_seg_idxs]

            q_split = np.log((self.error[phase_path][split_seg_idxs] /
                              self.previous_error[phase_path][split_parent_seg_idxs]) /
                             (seg_order[split_seg_idxs] / self.previous_gd[phase_path].transcription_order[
                                 split_parent_seg_idxs]) ** 2.5
                             ) / np.log((h[split_seg_idxs] / h_prev[split_parent_seg_idxs]) /
                                        (seg_order[split_seg_idxs] / self.previous_gd[phase_path].transcription_order[
                                            split_parent_seg_idxs]))

            q_split[q_split < 3] = 3
            q_split[np.isposinf(q_split)] = 3

            H[split_seg_idxs] = np.maximum(np.minimum(
                np.ceil((self.error[phase_path][split_seg_idxs] / phase.refine_options['tolerance']
                         ) ** (1 / q_split)), np.ceil(np.log(self.error[phase_path][split_seg_idxs] /
                                                             phase.refine_options['tolerance']) /
                                                      np.log(seg_order[split_seg_idxs]))),
                2*np.ones(split_seg_idxs.size))

            # reduce segment order where error is much below the tolerance
            if reduce_order_indx.size > 0:
                # compute normalization factor beta
                beta = {}
                for state_name, options in phase.state_options.items():
                    beta[state_name] = 0
                    for k in range(0, numseg):
                        beta_seg = np.max(np.abs(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))
                        if beta_seg > beta[state_name]:
                            beta[state_name] = beta_seg
                    beta[state_name] += 1

                for k in np.nditer(reduce_order_indx):
                    seg_size = {'radau-ps': seg_order[k] + 1, 'gauss-lobatto': seg_order[k]}
                    if seg_order[k] == phase.refine_options['min_order']:
                        continue
                    new_order_state = {}
                    new_order[k] = seg_order[k]
                    a = np.zeros((seg_size[gd.transcription], seg_size[gd.transcription]))
                    s, _ = lgr(seg_order[k], include_endpoint=True)
                    if gd.transcription == 'gauss-lobatto':
                        s, _ = lgl(seg_order[k])
                    for j in range(0, seg_size[gd.transcription]):
                        roots = s[s != s[j]]
                        Q = np.poly(roots)
                        a[:, j] = Q / np.polyval(Q, s[j])

                    for state_name, options in phase.state_options.items():
                        new_order_state[state_name] = seg_order[k]
                        b = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]

                        for i in range(seg_size[gd.transcription] - 1, phase.refine_options['min_order'], -1):
                            if np.min(np.abs(b[i])) / beta[state_name] < phase.refine_options['tolerance']/10 and \
                                    i - 1 < new_order_state[state_name]:
                                new_order_state[state_name] = i - 1
                            else:
                                new_order_state[state_name] = seg_order[k]
                        new_order[k] = max(new_order_state.values())

            # combine unnecessary segments
            merge_seg = np.zeros(numseg, dtype=bool)
            if check_comb_indx.size > 0:
                for k in np.nditer(check_comb_indx):
                    seg_size = {'radau-ps': seg_order[k] + 1, 'gauss-lobatto': seg_order[k]}
                    if merge_seg[k]:
                        continue

                    a = np.zeros((seg_size[gd.transcription], seg_size[gd.transcription]))
                    h_ = np.maximum(h[k], h[k + 1])
                    s, _ = lgr(new_order[k].astype(int), include_endpoint=True)
                    if gd.transcription == 'gauss-lobatto':
                        s, _ = lgl(new_order[k])
                    for j in range(0, seg_size[gd.transcription]):
                        roots = s[s != s[j]]
                        Q = np.poly(roots)
                        a[:, j] = Q / np.polyval(Q, s[j])

                    merge_seg[k + 1] = True

                    for state_name, options in phase.state_options.items():
                        beta = 1 + np.max(x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])
                        c = a @ x[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]
                        b = np.multiply(c.ravel(), np.array([(h_ / h[k]) ** l for l in
                                                             range(seg_size[gd.transcription])]))
                        b_hat = np.multiply(c.ravel(),
                                            np.array([(h_ / h[k + 1]) ** l for l in range(seg_size[gd.transcription])]))
                        err_val = np.dot(np.absolute(b - b_hat).ravel(),
                                         np.array([2 ** l for l in range(seg_size[gd.transcription])])) / beta

                        if err_val > phase.refine_options['tolerance']/10 and merge_seg[k + 1]:
                            merge_seg[k + 1] = False

            H[np.where(merge_seg)] = 0

            new_order = np.repeat(new_order, repeats=H)
            new_num_segments = int(np.sum(H))
            new_segment_ends = split_segments(gd.segment_ends, H)

            if gd.transcription == 'gauss-lobatto':
                new_order[new_order % 2 == 0] = new_order[new_order % 2 == 0] + 1

            self.parent_seg_map[phase_path] = np.zeros(new_num_segments, dtype=int)
            for i in range(1, new_num_segments):
                for j in range(1, numseg):
                    if new_segment_ends[i] == gd.segment_ends[j]:
                        self.parent_seg_map[phase_path][i] = int(j)
                        break
                    elif gd.segment_ends[j - 1] < new_segment_ends[i] < gd.segment_ends[j]:
                        self.parent_seg_map[phase_path][i] = int(j - 1)
                        break

            refine_results[phase_path]['new_order'] = new_order
            refine_results[phase_path]['new_num_segments'] = new_num_segments
            refine_results[phase_path]['new_segment_ends'] = new_segment_ends

            tx.options['order'] = new_order
            tx.options['num_segments'] = new_num_segments
            tx.options['segment_ends'] = new_segment_ends
            tx.init_grid()
            self.previous_x_dd[phase_path] = x_dd[phase_path].copy()
            self.previous_error[phase_path] = self.error[phase_path].copy()
            self.previous_gd[phase_path] = copy.deepcopy(gd)
