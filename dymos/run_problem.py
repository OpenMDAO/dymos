from .grid_refinement.ph_adaptive.ph_adaptive import PHAdaptive
from .phase.phase import Phase

import openmdao.api as om

def run_problem(problem, refine=True):
    problem.run_driver()

    if refine:

        phase_paths = find_phases(problem.model)

        for phase_path in phase_paths:
            phase = problem.model._get_subsystem(phase_path)
            print(phase)
            ph = PHAdaptive(phase)
            need_refine = ph.check_error()
            print(need_refine)
            if need_refine:
                new_order, new_node_num, new_segment_ends = ph.refine()
                #phase.options['transcription'] =



def find_phases(sys):
    phase_paths = []
    if isinstance(sys, Phase):
        phase_paths.append(sys.pathname)
    elif isinstance(sys, om.Group):
        for subsys in sys._loc_subsys_map:
            phase_paths.extend(find_phases(getattr(sys, subsys)))
    return phase_paths