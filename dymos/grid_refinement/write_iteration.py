"""
Utility for storing and outputting grid refinement information at each iteration
"""
import numpy as np


def write_error(f, iter_number, phases, refine_results):
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

    return


def write_refine_iter(f, iter_number, phases, refine_results):
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

    for phase_path, phase in phases.items():
        refine_data = refine_results[phase_path]
        refine_options = phase.refine_options

        f.write('    Phase: {}\n'.format(phase_path))
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
    return
