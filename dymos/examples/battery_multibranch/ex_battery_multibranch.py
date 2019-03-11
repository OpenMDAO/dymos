"""
Example that shows how to use multiple phases in Dymos to model failure of a battery cell
in a simple electrical system.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, IndepVarComp, DirectSolver

from dymos import Phase, Trajectory
from dymos.utils.lgl import lgl

from battery_multibranch_ode import BatteryODE


def run_example(optimizer='SLSQP', transcription='gauss-lobatto'):

    prob = Problem(model=Group())

    opt = prob.driver = pyOptSparseDriver()

    opt.options['optimizer'] = optimizer
    opt.options['dynamic_simul_derivs'] = True
    if optimizer == 'SNOPT':
        opt.opt_settings['Major iterations limit'] = 1000
        opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
        opt.opt_settings['Major optimality tolerance'] = 1.0E-6
        opt.opt_settings["Linesearch tolerance"] = 0.10
        opt.opt_settings['iSumm'] = 6

    num_seg = 5
    seg_ends, _ = lgl(num_seg + 1)

    traj = prob.model.add_subsystem('traj', Trajectory())

    # First phase: normal operation.

    phase0 = Phase(transcription,
                   ode_class=BatteryODE,
                   num_segments=num_seg,
                   segment_ends=seg_ends,
                   transcription_order=5,
                   compressed=False)

    traj_p0 = traj.add_phase('phase0', phase0)

    traj_p0.set_time_options(fix_initial=True, fix_duration=True)
    traj_p0.set_state_options('state_of_charge', fix_initial=True, fix_final=False)

    # Second phase: normal operation.

    phase1 = Phase(transcription,
                   ode_class=BatteryODE,
                   num_segments=num_seg,
                   segment_ends=seg_ends,
                   transcription_order=5,
                   compressed=False)

    traj_p1 = traj.add_phase('phase1', phase1)

    traj_p1.set_time_options(fix_initial=False, fix_duration=True)
    traj_p1.set_state_options('state_of_charge', fix_initial=False, fix_final=False)
    traj_p1.add_objective('time', loc='final')

    # Second phase, but with battery failure.

    phase1_bfail = Phase(transcription,
                         ode_class=BatteryODE,
                         num_segments=num_seg,
                         segment_ends=seg_ends,
                         transcription_order=5,
                         ode_init_kwargs={'num_battery': 2},
                         compressed=False)

    traj_p1_bfail = traj.add_phase('phase1_bfail', phase1_bfail)

    traj_p1_bfail.set_time_options(fix_initial=False, fix_duration=True)
    traj_p1_bfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False)

    # Second phase, but with motor failure.

    phase1_mfail = Phase(transcription,
                         ode_class=BatteryODE,
                         num_segments=num_seg,
                         segment_ends=seg_ends,
                         transcription_order=5,
                         ode_init_kwargs={'num_motor': 2},
                         compressed=False)

    traj_p1_mfail = traj.add_phase('phase1_mfail', phase1_mfail)

    traj_p1_mfail.set_time_options(fix_initial=False, fix_duration=True)
    traj_p1_mfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False)

    traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'])
    traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'])
    traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'])

    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)

    prob.setup()

    prob['traj.phases.phase0.time_extents.t_initial'] = 0
    prob['traj.phases.phase0.time_extents.t_duration'] = 1.0*3600

    prob['traj.phases.phase1.time_extents.t_initial'] = 1.0*3600
    prob['traj.phases.phase1.time_extents.t_duration'] = 1.0*3600

    prob['traj.phases.phase1_bfail.time_extents.t_initial'] = 1.0*3600
    prob['traj.phases.phase1_bfail.time_extents.t_duration'] = 1.0*3600

    prob['traj.phases.phase1_mfail.time_extents.t_initial'] = 1.0*3600
    prob['traj.phases.phase1_mfail.time_extents.t_duration'] = 1.0*3600

    prob.set_solver_print(level=0)
    prob.run_driver()

    return prob


if __name__ == '__main__':
    prob = run_example(optimizer='SNOPT', transcription='radau-ps')

    plot = True
    if plot:
        import matplotlib.pyplot as plt
        traj = prob.model.traj

        t0 = prob['traj.phases.phase0.time.time']/3600
        t1 = prob['traj.phases.phase1.time.time']/3600
        t1b = prob['traj.phases.phase1_bfail.time.time']/3600
        t1m = prob['traj.phases.phase1_mfail.time.time']/3600
        soc0 = prob['traj.phases.phase0.indep_states.states:state_of_charge']
        soc1 = prob['traj.phases.phase1.indep_states.states:state_of_charge']
        soc1b = prob['traj.phases.phase1_bfail.indep_states.states:state_of_charge']
        soc1m = prob['traj.phases.phase1_mfail.indep_states.states:state_of_charge']

        plt.subplot(2, 2, 1)
        plt.plot(t0, soc0, 'b')
        plt.plot(t1, soc1, 'b')
        plt.plot(t1b, soc1b, 'r')
        plt.plot(t1m, soc1m, 'c')
        plt.xlabel('Time (hour)')
        plt.ylabel('State of Charge (percent)')

        V_oc0 = prob['traj.phases.phase0.rhs_all.battery.V_oc']
        V_oc1 = prob['traj.phases.phase1.rhs_all.battery.V_oc']
        V_oc1b = prob['traj.phases.phase1_bfail.rhs_all.battery.V_oc']
        V_oc1m = prob['traj.phases.phase1_mfail.rhs_all.battery.V_oc']

        plt.subplot(2, 2, 2)
        plt.plot(t0, V_oc0, 'b')
        plt.plot(t1, V_oc1, 'b')
        plt.plot(t1b, V_oc1b, 'r')
        plt.plot(t1m, V_oc1m, 'c')
        plt.xlabel('Time (hour)')
        plt.ylabel('Open Circuit Voltage (V)')

        V_pack0 = prob['traj.phases.phase0.rhs_all.battery.V_pack']
        V_pack1 = prob['traj.phases.phase1.rhs_all.battery.V_pack']
        V_pack1b = prob['traj.phases.phase1_bfail.rhs_all.battery.V_pack']
        V_pack1m = prob['traj.phases.phase1_mfail.rhs_all.battery.V_pack']

        plt.subplot(2, 2, 3)
        plt.plot(t0, V_pack0, 'b')
        plt.plot(t1, V_pack1, 'b')
        plt.plot(t1b, V_pack1b, 'r')
        plt.plot(t1m, V_pack1m, 'c')
        plt.xlabel('Time (hour)')
        plt.ylabel('Terminal Voltage (V)')

        I_Li0 = prob['traj.phases.phase0.rhs_all.pwr_balance.I_Li']
        I_Li1 = prob['traj.phases.phase1.rhs_all.pwr_balance.I_Li']
        I_Li1b = prob['traj.phases.phase1_bfail.rhs_all.pwr_balance.I_Li']
        I_Li1m = prob['traj.phases.phase1_mfail.rhs_all.pwr_balance.I_Li']

        plt.subplot(2, 2, 4)
        plt.plot(t0, I_Li0, 'b')
        plt.plot(t1, I_Li1, 'b')
        plt.plot(t1b, I_Li1b, 'r')
        plt.plot(t1m, I_Li1m, 'c')
        plt.xlabel('Time (hour)')
        plt.ylabel('Line Current (A)')

        plt.show()

    print('done')
