"""
Utility for performing grid refinement on each phase.
"""
from .ph_adaptive.ph_adaptive import PHAdaptive
from .hp_adaptive.hp_adaptive import HPAdaptive
from .write_iteration import write_error, write_refine_iter

from dymos.grid_refinement.error_estimation import check_error
from dymos.load_case import load_case, find_phases

import numpy as np
import sys


def _refine_iter(problem, refine_iteration_limit=0, refine_method='hp'):
    """
    This function performs grid refinement for a phases in which solve_segments is true.

    Parameters
    ----------
    problem : om.Problem
        The OpenMDAO problem object to be run.
    refine_method : String
        The choice of refinement algorithm to use for grid refinement
    refine_iteration_limit : int
        The number of passes through the grid refinement algorithm to be made.
    """
    phases = find_phases(problem.model)
    refinement_methods = {'hp': HPAdaptive, 'ph': PHAdaptive}

    if refine_iteration_limit > 0:
        out_file = 'grid_refinement.out'

        ref = refinement_methods[refine_method](phases)
        with open(out_file, 'w+') as f:
            for i in range(refine_iteration_limit):
                refine_results = check_error(phases)

                refined_phases = [phase_path for phase_path in refine_results if
                                  phases[phase_path].refine_options['refine'] and
                                  np.any(refine_results[phase_path]['need_refinement'])]

                for stream in f, sys.stdout:
                    write_error(stream, i, phases, refine_results)

                if not refined_phases:
                    break

                ref.refine(refine_results, i)

                for stream in f, sys.stdout:
                    write_refine_iter(stream, i, phases, refine_results)

                prev_soln = {'inputs': problem.model.list_inputs(out_stream=None, units=True, prom_name=True),
                             'outputs': problem.model.list_outputs(out_stream=None, units=True, prom_name=True)}

                problem.setup()

                load_case(problem, prev_soln)

                problem.run_driver()

                for stream in [f, sys.stdout]:
                    if i == refine_iteration_limit-1:
                        print('Iteration limit exceeded. Unable to satisfy specified tolerance', file=stream)
                    else:
                        print('Successfully completed grid refinement.', file=stream)
                print(50 * '=')
