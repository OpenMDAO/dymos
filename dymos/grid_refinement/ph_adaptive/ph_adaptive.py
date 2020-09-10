from ...phase.phase import Phase

import numpy as np


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


class PHAdaptive:
    """
    Grid refinement object for the p-then-h grid refinement algorithm

    The error on a solved phase is evaluated. If error exceeds chosen tolerance, the grid is refined following the
    p-then-h refinement algorithm.
    Patterson, M. A., Hager, W. W., and Rao. A. V., “A ph Mesh Refinement Method for Optimal Control”,
    Optimal Control Applications and Methods, Vol. 36, No. 4, July - August 2015, pp. 398 - 421. DOI: 10.1002/oca2114

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

    def refine(self, refine_results, iter_number):
        """
        Compute the order, number of nodes, and segment ends required for the new grid
        and assigns them to the transcription of each phase.

        Parameters
        ----------
        iter_number: int
            An integer value representing the iteration of the grid refinement

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

            error = refine_data['max_rel_error']
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
