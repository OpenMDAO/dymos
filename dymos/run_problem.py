from .grid_refinement.ph_adaptive.ph_adaptive import PHAdaptive
from .phase.phase import Phase

import numpy as np
import openmdao.api as om


def run_problem(problem, refine=True):
    problem.run_driver()

    if refine:

        phases = {phase_path: problem.model._get_subsystem(phase_path)
                  for phase_path in find_phases(problem.model)}

        ph = PHAdaptive(phases)

        for i in range(ph.iteration_limit):
            need_refine = ph.check_error()

            import copy
            prev_soln = {}
            prev_soln['inputs'] = copy.deepcopy(problem.model.list_inputs(out_stream=None, units=True))
            prev_soln['outputs'] = copy.deepcopy(problem.model.list_outputs(out_stream=None, units=True))

            ph.refine(need_refine)

            problem.setup()

            re_interpolate_solution(problem, phases, previous_solution=prev_soln)

            problem.run_driver()

        exit(0)

        # for i in range ph.iteration_limit:

        # need_refine = {}

        # #
        # # Determine the phase/segments which need refinement
        # #
        # phase_paths = find_phases(problem.model)
        # print(phase_paths)
        # for phase_path in phase_paths:
        #     phase = problem.model._get_subsystem(phase_path)
        #     ph = PHAdaptive(phase)
        #     need_refine[phase_path] = ph.check_error()
        #     M = 0
        #     while np.any(need_refine[phase_path]) or M >= ph.iteration_limit:
        #         new_order, new_num_segments, new_segment_ends = ph.refine(need_refine[phase_path])
        #         T = phase.options['transcription']
        #
        #         T.options['order'] = new_order
        #         T.options['num_segments'] = new_num_segments
        #         T.options['segment_ends'] = new_segment_ends

        #
        # If refinement was needed, setup a new problem
        #
        # problem.setup()
        #
        # re_interpolate_solution(problem, phase, phase_path)
        #
        # #
        # # Set the initial values for the newly refined phases
        # #
        # for phase_path in phase_paths:
        #     phase = problem.model._get_subsystem(phase_path)
        #     ph = PHAdaptive(phase)
        #     need_refine[phase_path] = ph.check_error()
        #     M = 0
        #     while np.any(need_refine[phase_path]) or M >= ph.iteration_limit:
        #         new_order, new_num_segments, new_segment_ends = ph.refine(need_refine[phase_path])
        #         T = phase.options['transcription']
        #
        #         T.options['order'] = new_order
        #         T.options['num_segments'] = new_num_segments
        #         T.options['segment_ends'] = new_segment_ends
        #
        #
        #         # interpolate previous solution into new grid
        #         re_interpolate_solution(problem, phase, phase_path)
        #
        #         problem.run_driver()
        #         need_refine = ph.check_error()
        #         M += 1


def find_phases(sys):
    phase_paths = []
    if isinstance(sys, Phase):
        phase_paths.append(sys.pathname)
    elif isinstance(sys, om.Group):
        for subsys in sys._loc_subsys_map:
            phase_paths.extend(find_phases(getattr(sys, subsys)))
    return phase_paths


def re_interpolate_solution(problem, phases, previous_solution):

    phase_paths = phases.keys()

    prev_ip_dict = {k: v['value'] for k, v in previous_solution['inputs']}
    prev_op_dict = {k: v['value'] for k, v in previous_solution['outputs']}

    # prom_to_abs_ip_map = problem.model._var_allprocs_prom2abs_list['input']
    # prom_to_abs_op_map = problem.model._var_allprocs_prom2abs_list['output']

    abs_to_prom_ip_map = {}
    for prom_name, abs_names in problem.model._var_allprocs_prom2abs_list['input'].items():
        for abs_name in abs_names:
            abs_to_prom_ip_map[abs_name] = prom_name

    abs_to_prom_op_map = {}
    for prom_name, abs_names in problem.model._var_allprocs_prom2abs_list['output'].items():
        for abs_name in abs_names:
            abs_to_prom_op_map[abs_name] = prom_name

    for phase_path, phase in phases.items():
        prom_to_abs_ip_map = phase._var_allprocs_prom2abs_list['input']
        prom_to_abs_op_map = phase._var_allprocs_prom2abs_list['output']

        ti_abs_name = prom_to_abs_op_map['t_initial'][0]
        ti_prom_name = abs_to_prom_op_map[f'{phase_path}.time_extents.t_initial']
        t_initial = prev_op_dict[ti_abs_name]

        td_abs_name = prom_to_abs_op_map['t_duration'][0]
        td_prom_name = abs_to_prom_op_map[f'{phase_path}.time_extents.t_duration']
        t_duration = prev_op_dict[td_abs_name]

        prev_time = prev_op_dict[f'{phase_path}.time.time']

        problem.set_val(ti_prom_name, t_initial)
        problem.set_val(td_prom_name, t_duration)

        for state_name, options in phase.state_options.items():
            state_abs_name = f'{phase_path}.indep_states.states:{state_name}'
            prev_state_soln_abs_name = f'{phase_path}.timeseries.states:{state_name}'
            state_prom_name = abs_to_prom_op_map[state_abs_name]
            prev_state_val = prev_op_dict[prev_state_soln_abs_name]
            problem.set_val(state_prom_name, phase.interpolate(xs=prev_time, ys=prev_state_val, nodes='state_input'))

        print(abs_to_prom_op_map.keys())

        for control_name, options in phase.control_options.items():
            control_abs_name = f'{phase_path}.control_group.indep_controls.controls:{control_name}'
            prev_control_soln_abs_name = f'{phase_path}.timeseries.controls:{control_name}'
            control_prom_name = abs_to_prom_op_map[control_abs_name]
            prev_control_val = prev_op_dict[prev_control_soln_abs_name]
            problem.set_val(control_prom_name, phase.interpolate(xs=prev_time, ys=prev_control_val, nodes='control_input'))

    #
    #
    # abs_to_prom_ip_map = {v: k for k, v in prom_to_abs_ip_map.items()}
    # abs_to_prom_op_map = {v: k for k, v in prom_to_abs_op_map.items()}

    # First reassign anything that wasn't in a regridded phase
    # for ip_name, value in prev_ip_dict.items():
    #     prom_name = abs_to_prom_ip_map[ip_name]
    #     problem.set_val(prom_name, value)
    #
    # for op_name, value in prev_op_dict.items():
    #     prom_name = abs_to_prom_op_map[op_name]
    #     print(prom_name, len(value), len(problem.get_val(prom_name)))
    #     problem[prom_name][...] = value
    #     print(value.T)
    #     print()
    #     print(problem[prom_name].T)

    # for phase_path, phase in phases.items():
    #
    # problem.set_val('traj.phase0.states:x', phase.interpolate(ys=[1.5, 1], nodes='state_input'))
    # problem.set_val('traj.phase0.states:xL', phase.interpolate(ys=[0, 1], nodes='state_input'))
    # problem.set_val('traj.phase0.t_initial', 0)
    # problem.set_val('traj.phase0.t_duration', 100)
    # problem.set_val('traj.phase0.controls:u', phase.interpolate(ys=[-0.6, 2.4], nodes='control_input'))
    #
    # # phase_items = phase.list_outputs(explicit=True, implicit=True, values=False, prom_name=True)
    # # for item in phase_items:
    # #     problem.set_val(item, 0)
