from .grid_refinement.ph_adaptive.ph_adaptive import PHAdaptive
from .phase.phase import Phase

import numpy as np
import openmdao.api as om


def run_problem(problem, refine=True):
    problem.run_driver()

    if refine:

        phase_paths = find_phases(problem.model)

        for phase_path in phase_paths:
            phase = problem.model._get_subsystem(phase_path)
            ph = PHAdaptive(phase)
            need_refine = ph.check_error()
            M = 0
            while np.any(need_refine) or M >= ph.iteration_limit:
                new_order, new_num_segments, new_segment_ends = ph.refine(need_refine)
                T = phase.options['transcription']
                T['order'] = new_order
                T['num_segments'] = new_num_segments
                T['segment_ends'] = new_segment_ends
                exit(0)
                need_refine = ph.check_error()
                M += 1


def find_phases(sys):
    phase_paths = []
    if isinstance(sys, Phase):
        phase_paths.append(sys.pathname)
    elif isinstance(sys, om.Group):
        for subsys in sys._loc_subsys_map:
            phase_paths.extend(find_phases(getattr(sys, subsys)))
    return phase_paths
