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
            while np.any(need_refine):
                new_order, new_node_num, new_segment_ends = ph.refine()
                T = phase.options['transcription']
                need_refine = ph.check_error()


def find_phases(sys):
    phase_paths = []
    if isinstance(sys, Phase):
        phase_paths.append(sys.pathname)
    elif isinstance(sys, om.Group):
        for subsys in sys._loc_subsys_map:
            phase_paths.extend(find_phases(getattr(sys, subsys)))
    return phase_paths
