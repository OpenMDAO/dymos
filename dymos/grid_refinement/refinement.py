"""
Utility for performing grid refinement on each phase.
"""
from .ph_adaptive.ph_adaptive import PHAdaptive
from .hp_adaptive.hp_adaptive import HPAdaptive
from .write_iteration import write_error, write_refine_iter

from dymos.grid_refinement.error_estimation import check_error
from dymos.phase import Phase

from copy import deepcopy

import numpy as np
import openmdao.api as om
import sys


def find_phases(sys):
    """
    Find all instances of Dymos Phases within the given system, and returns them as a dictionary.

    Parameters
    ----------
    sys : om.Group
        The OpenMDAO Group to be searched for Dymos Phases.

    Returns
    -------
    dict
        A dictionary mapping the absolute path of each Phase object in the given group to each
        Phase object.
    """
    phase_paths = {}
    if isinstance(sys, Phase):
        phase_paths[sys.pathname] = sys
    elif isinstance(sys, om.Group):
        for sub in sys.system_iter(recurse=False):
            phase_paths.update(find_phases(sub))
    return phase_paths


def _refine_iter(problem, refine_iteration_limit=0, refine_method='hp', case_prefix=None, reset_iter_counts=True):
    """
    Perform grid refinement for phases in which solve_segments is true.

    Parameters
    ----------
    problem : om.Problem
        The OpenMDAO problem object to be run.
    refine_method : String
        The choice of refinement algorithm to use for grid refinement
    refine_iteration_limit : int
        The number of passes through the grid refinement algorithm to be made.
    case_prefix : str or None
        Prefix to prepend to coordinates when recording.
    reset_iter_counts : bool
        If True and model has been run previously, reset all iteration counters.
    """
    phases = find_phases(problem.model)
    refinement_methods = {'hp': HPAdaptive, 'ph': PHAdaptive}
    _case_prefix = '' if case_prefix is None else f'{case_prefix}_'

    case_prefix = f'{_case_prefix}{refine_method}_0_'
    failed = problem.run_driver(case_prefix=case_prefix if refine_iteration_limit > 0 else _case_prefix,
                                reset_iter_counts=reset_iter_counts)

    if refine_iteration_limit > 0:
        out_file = 'grid_refinement.out'

        ref = refinement_methods[refine_method](phases)
        with open(out_file, 'w+') as f:
            for i in range(1, refine_iteration_limit + 1):
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

                prev_soln = {
                    'inputs': problem.model.list_inputs(out_stream=None, return_format='dict',
                                                        units=True, prom_name=True),
                    'outputs': problem.model.list_outputs(out_stream=None, return_format='dict',
                                                          units=True, prom_name=True)
                }

                # Setup resets the value of every variable. Keep the values of the independent
                # variables, so that those whose shape is unchanged can be restored after setup.
                prev_indeps = {name: (deepcopy(meta['val']), meta.get('units'))
                               for name, meta in dict(problem.list_indep_vars(out_stream=None)).items()}

                problem.setup()
                problem.final_setup()

                for name, meta in dict(problem.list_indep_vars(out_stream=None)).items():
                    if name in prev_indeps:
                        val, units = prev_indeps[name]
                        if np.shape(val) == np.shape(meta['val']):
                            problem.set_val(name, val, units=units)

                # Interpolate the previous solution onto the new grid of the refined phases.
                for phase_path in refined_phases:
                    phs = problem.model._get_subsystem(phase_path)
                    phs.load_case(prev_soln)

                failed = problem.run_driver(case_prefix=f'{_case_prefix}{refine_method}_{i}_')

            for stream in [f, sys.stdout]:
                if i == refine_iteration_limit - 1:
                    print('Iteration limit exceeded. Unable to satisfy specified tolerance', file=stream)
                    failed = True
                else:
                    print('Successfully completed grid refinement.', file=stream)
            print(50 * '=')

    return failed
