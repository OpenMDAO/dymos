import numpy as np


def split_segments(old_seg_ends, B):
    """
    Compute the new segment ends for the refined grid by splitting the necessary segments.

    Parameters
    ----------
    old_seg_ends : ndarray
        Segment ends of grid on which the problem was solved.
    B : ndarray of ints
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


class PHAdaptive:
    """
    Grid refinement object for the p-then-h grid refinement algorithm.

    The error on a solved phase is evaluated. If error exceeds chosen tolerance, the grid is refined following the
    p-then-h refinement algorithm.
    Patterson, M. A., Hager, W. W., and Rao. A. V., “A ph Mesh Refinement Method for Optimal Control”,
    Optimal Control Applications and Methods, Vol. 36, No. 4, July - August 2015, pp. 398 - 421. DOI: 10.1002/oca2114

    Parameters
    ----------
    phases : Phase
        The Phase object representing the solved phase.
    """

    def __init__(self, phases):
        self.phases = phases
        self.error = {}

    def refine(self, refine_results, iter_number):
        """
        Refine the grid.

        Compute the order, number of nodes, and segment ends required for the new grid
        and assigns them to the transcription of each phase.

        Parameters
        ----------
        refine_results : dict
            A dictionary where each key is the path to a phase in the problem, and the
            associated value are various properties of that phase needed by the refinement
            algorithm.  refine_results is returned by check_error.  This method modifies it
            in place, adding the new_num_segments, new_order, and new_segment_ends.
        iter_number : int
            Iteration number of the grid refinement.

        Returns
        -------
        dict
            A dictionary of phase paths : phases which were refined.
        """
        for phase_path, phase_refinement_results in refine_results.items():
            phase = self.phases[phase_path]
            tx = phase.options['transcription']
            gd = tx.grid_data

            need_refine = phase_refinement_results['need_refinement']
            if not phase.refine_options['refine'] or not np.any(need_refine):
                refine_results[phase_path]['new_order'] = gd.transcription_order
                refine_results[phase_path]['new_num_segments'] = gd.num_segments
                refine_results[phase_path]['new_segment_ends'] = gd.segment_ends
                continue

            # Refinement is needed
            gd = phase.options['transcription'].grid_data
            numseg = gd.num_segments

            refine_seg_idxs = np.where(need_refine)
            P = np.zeros(numseg)

            max_rel_error = refine_results[phase_path]['max_rel_error'][refine_seg_idxs]
            tol = phase.refine_options['tolerance']
            order = gd.transcription_order[refine_seg_idxs]

            P[refine_seg_idxs] = np.log(max_rel_error / tol) / np.log(order)
            P = np.ceil(P).astype(int)

            if gd.transcription == 'gauss-lobatto':
                odd_idxs = np.where(P % 2 != 0)
                P[odd_idxs] += 1

            new_order = gd.transcription_order + P
            B = np.ones(numseg, dtype=int)

            raise_order_idxs = np.where(gd.transcription_order + P <= phase.refine_options['max_order'])
            split_seg_idxs = np.where(gd.transcription_order + P > phase.refine_options['max_order'])

            new_order[raise_order_idxs] = gd.transcription_order[raise_order_idxs] + P[raise_order_idxs]
            new_order[split_seg_idxs] = phase.refine_options['min_order']

            B[split_seg_idxs] = np.around((gd.transcription_order[split_seg_idxs] +
                                           P[split_seg_idxs]) / phase.refine_options['min_order']).astype(int)

            new_order = np.repeat(new_order, repeats=B)
            new_num_segments = int(np.sum(B))
            new_segment_ends = split_segments(gd.segment_ends, B)

            refine_results[phase_path]['new_order'] = new_order
            refine_results[phase_path]['new_num_segments'] = new_num_segments
            refine_results[phase_path]['new_segment_ends'] = new_segment_ends

            tx.options['order'] = new_order
            tx.options['num_segments'] = new_num_segments
            tx.options['segment_ends'] = new_segment_ends
            tx.init_grid()
