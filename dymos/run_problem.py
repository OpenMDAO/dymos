from .grid_refinement.ph_adaptive.ph_adaptive import PHAdaptive
from .phase.phase import Phase

import numpy as np
import openmdao.api as om


def run_problem(problem, refine=True):
    problem.run_driver()

    if refine:

        phase_paths = find_phases(problem.model)
        print(phase_paths)
        for phase_path in phase_paths:
            phase = problem.model._get_subsystem(phase_path)
            print(phase_path)
            ph = PHAdaptive(phase)
            need_refine = ph.check_error()
            M = 0
            while np.any(need_refine) or M >= ph.iteration_limit:
                new_order, new_num_segments, new_segment_ends = ph.refine(need_refine)
                T = phase.options['transcription']

                T.options['order'] = new_order
                T.options['num_segments'] = new_num_segments
                T.options['segment_ends'] = new_segment_ends
                problem.setup()

                # interpolate previous solution into new grid
                re_interpolate_solution(problem, phase, phase_path)

                problem.run_driver()
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


def re_interpolate_solution(problem, phase, phase_path):
    problem.set_val('traj.phase0.states:x', phase.interpolate(ys=[1.5, 1], nodes='state_input'))
    problem.set_val('traj.phase0.states:xL', phase.interpolate(ys=[0, 1], nodes='state_input'))
    problem.set_val('traj.phase0.t_initial', 0)
    problem.set_val('traj.phase0.t_duration', 100)
    problem.set_val('traj.phase0.controls:u', phase.interpolate(ys=[-0.6, 2.4], nodes='control_input'))

    phase_items = phase.list_outputs(explicit=True, implicit=True, values=False, prom_name=True)
    for item in phase_items:
        problem.set_val(item, 0)
