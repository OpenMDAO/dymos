"""
Example that shows how to use multiple phases in Dymos to model failure of a battery cell
in a simple electrical system.
"""
from __future__ import division, print_function, absolute_import

import unittest

from openmdao.api import Problem, Group

from dymos import  Trajectory, Phase, GaussLobatto, Radau, RungeKutta

from dymos.examples.battery_multibranch.battery_multibranch_ode import BatteryODE, \
    BatteryODEStaticGearboxMotorPower


class TestBatteryRKIVP(unittest.TestCase):

    def test_dynamic_input_params(self):
        prob = Problem(model=Group())

        num_seg = 5

        traj = prob.model.add_subsystem('traj', Trajectory())

        # First phase: normal operation.
        # NOTE: using RK4 integration here

        P_DEMAND = 2.0


        phase0 = Phase(ode_class=BatteryODE, transcription=RungeKutta(num_segments=200))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.set_state_options('state_of_charge', fix_initial=True, fix_final=False)
        phase0.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase0.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase0.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase0.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase0', phase0)

        # Second phase: normal operation.

        transcription = Radau(num_segments=5, order=5, compressed=True)
        phase1 =  Phase(ode_class=BatteryODE, transcription=transcription)
        phase1.set_time_options(fix_initial=False, fix_duration=True)
        phase1.set_state_options('state_of_charge', fix_initial=False, fix_final=False, solve_segments=True)
        phase1.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase1.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase1.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase1.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase1', phase1)

        # Second phase, but with battery failure.

        phase1_bfail = Phase(ode_class=BatteryODE, ode_init_kwargs={'num_battery': 2}, 
                         transcription=transcription)
        phase1_bfail.set_time_options(fix_initial=False, fix_duration=True)
        phase1_bfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False, solve_segments=True)
        phase1_bfail.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase1_bfail.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase1_bfail.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase1_bfail.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase1_bfail', phase1_bfail)

        # Second phase, but with motor failure.

        phase1_mfail = Phase(ode_class=BatteryODE, ode_init_kwargs={'num_motor': 2}, 
                         transcription=transcription)
        phase1_mfail.set_time_options(fix_initial=False, fix_duration=True)
        phase1_mfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False, solve_segments=True)
        phase1_mfail.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase1_mfail.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase1_mfail.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase1_mfail.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase1_mfail', phase1_mfail)

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'], connected=True)
        traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'], connected=True)
        traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'], connected=True)

        # prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.setup()
        prob.final_setup()

        prob['traj.phases.phase0.time_extents.t_initial'] = 0
        prob['traj.phases.phase0.time_extents.t_duration'] = 1.0*3600

        # prob['traj.phases.phase1.time_extents.t_initial'] = 1.0*3600
        prob['traj.phases.phase1.time_extents.t_duration'] = 1.0*3600

        # prob['traj.phases.phase1_bfail.time_extents.t_initial'] = 1.0*3600
        prob['traj.phases.phase1_bfail.time_extents.t_duration'] = 1.0*3600

        # prob['traj.phases.phase1_mfail.time_extents.t_initial'] = 1.0*3600
        prob['traj.phases.phase1_mfail.time_extents.t_duration'] = 1.0*3600

        prob.set_solver_print(level=0)
        prob.run_model()

        plot = True
        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            traj = prob.model.traj

            t0 = prob['traj.phase0.timeseries.time']
            t1 = prob['traj.phase1.timeseries.time']
            t1b = prob['traj.phase1_bfail.timeseries.time']
            t1m = prob['traj.phase1_mfail.timeseries.time']
            soc0 = prob['traj.phase0.timeseries.states:state_of_charge']
            soc1 = prob['traj.phase1.timeseries.states:state_of_charge']
            soc1b = prob['traj.phase1_bfail.timeseries.states:state_of_charge']
            soc1m = prob['traj.phase1_mfail.timeseries.states:state_of_charge']

            plt.subplot(2, 2, 1)
            plt.plot(t0, soc0, 'b')
            plt.plot(t1, soc1, 'b')
            plt.plot(t1b, soc1b, 'r')
            plt.plot(t1m, soc1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('State of Charge (percent)')

            V_oc0 = prob['traj.phase0.timeseries.V_oc']
            V_oc1 = prob['traj.phase1.timeseries.V_oc']
            V_oc1b = prob['traj.phase1_bfail.timeseries.V_oc']
            V_oc1m = prob['traj.phase1_mfail.timeseries.V_oc']

            plt.subplot(2, 2, 2)
            plt.plot(t0, V_oc0, 'b')
            plt.plot(t1, V_oc1, 'b')
            plt.plot(t1b, V_oc1b, 'r')
            plt.plot(t1m, V_oc1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('Open Circuit Voltage (V)')

            V_pack0 = prob['traj.phase0.timeseries.V_pack']
            V_pack1 = prob['traj.phase1.timeseries.V_pack']
            V_pack1b = prob['traj.phase1_bfail.timeseries.V_pack']
            V_pack1m = prob['traj.phase1_mfail.timeseries.V_pack']

            plt.subplot(2, 2, 3)
            plt.plot(t0, V_pack0, 'b')
            plt.plot(t1, V_pack1, 'b')
            plt.plot(t1b, V_pack1b, 'r')
            plt.plot(t1m, V_pack1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('Terminal Voltage (V)')

            I_Li0 = prob['traj.phase0.timeseries.I_Li']
            I_Li1 = prob['traj.phase1.timeseries.I_Li']
            I_Li1b = prob['traj.phase1_bfail.timeseries.I_Li']
            I_Li1m = prob['traj.phase1_mfail.timeseries.I_Li']

            plt.subplot(2, 2, 4)
            plt.plot(t0, I_Li0, 'b')
            plt.plot(t1, I_Li1, 'b')
            plt.plot(t1b, I_Li1b, 'r')
            plt.plot(t1m, I_Li1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('Line Current (A)')

            plt.show()

    def test_static_input_params(self):
        prob = Problem(model=Group())

        num_seg = 5

        traj = prob.model.add_subsystem('traj', Trajectory())

        # First phase: normal operation.
        # NOTE: using RK4 integration here

        P_DEMAND = 2.0

        phase0 = Phase(ode_class=BatteryODE, transcription=RungeKutta(num_segments=200))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.set_state_options('state_of_charge', fix_initial=True, fix_final=False)
        phase0.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase0.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase0.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase0.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase0', phase0)

        # Second phase: normal operation.

        transcription = Radau(num_segments=5, order=5, compressed=True)
        phase1 =  Phase(ode_class=BatteryODE, transcription=transcription)
        phase1.set_time_options(fix_initial=False, fix_duration=True)
        phase1.set_state_options('state_of_charge', fix_initial=False, fix_final=False,
                                 solve_segments=True)
        phase1.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase1.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase1.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase1.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase1', phase1)

        # Second phase, but with battery failure.

        phase1_bfail = Phase(ode_class=BatteryODE, ode_init_kwargs={'num_battery': 2}, 
                         transcription=transcription)
        phase1_bfail.set_time_options(fix_initial=False, fix_duration=True)
        phase1_bfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False,
                                       solve_segments=True)
        phase1_bfail.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase1_bfail.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase1_bfail.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase1_bfail.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase1_bfail', phase1_bfail)

        # Second phase, but with motor failure.

        phase1_mfail = Phase(ode_class=BatteryODE, ode_init_kwargs={'num_motor': 2}, 
                         transcription=transcription)
        phase1_mfail.set_time_options(fix_initial=False, fix_duration=True)
        phase1_mfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False,
                                       solve_segments=True)
        phase1_mfail.add_timeseries_output('battery.V_oc', output_name='V_oc', units='V')
        phase1_mfail.add_timeseries_output('battery.V_pack', output_name='V_pack', units='V')
        phase1_mfail.add_timeseries_output('pwr_balance.I_Li', output_name='I_Li', units='A')
        phase1_mfail.add_input_parameter('P_demand', val=P_DEMAND, units='W')
        traj.add_phase('phase1_mfail', phase1_mfail)

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'],
                         connected=True)

        # prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.setup()
        prob.final_setup()

        prob['traj.phases.phase0.time_extents.t_initial'] = 0
        prob['traj.phases.phase0.time_extents.t_duration'] = 1.0 * 3600

        # prob['traj.phases.phase1.time_extents.t_initial'] = 1.0*3600
        prob['traj.phases.phase1.time_extents.t_duration'] = 1.0 * 3600

        # prob['traj.phases.phase1_bfail.time_extents.t_initial'] = 1.0*3600
        prob['traj.phases.phase1_bfail.time_extents.t_duration'] = 1.0 * 3600

        # prob['traj.phases.phase1_mfail.time_extents.t_initial'] = 1.0*3600
        prob['traj.phases.phase1_mfail.time_extents.t_duration'] = 1.0 * 3600

        prob.set_solver_print(level=0)
        prob.run_model()

        plot = True
        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            traj = prob.model.traj

            t0 = prob['traj.phase0.timeseries.time']
            t1 = prob['traj.phase1.timeseries.time']
            t1b = prob['traj.phase1_bfail.timeseries.time']
            t1m = prob['traj.phase1_mfail.timeseries.time']
            soc0 = prob['traj.phase0.timeseries.states:state_of_charge']
            soc1 = prob['traj.phase1.timeseries.states:state_of_charge']
            soc1b = prob['traj.phase1_bfail.timeseries.states:state_of_charge']
            soc1m = prob['traj.phase1_mfail.timeseries.states:state_of_charge']

            plt.subplot(2, 2, 1)
            plt.plot(t0, soc0, 'b')
            plt.plot(t1, soc1, 'b')
            plt.plot(t1b, soc1b, 'r')
            plt.plot(t1m, soc1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('State of Charge (percent)')

            V_oc0 = prob['traj.phase0.timeseries.V_oc']
            V_oc1 = prob['traj.phase1.timeseries.V_oc']
            V_oc1b = prob['traj.phase1_bfail.timeseries.V_oc']
            V_oc1m = prob['traj.phase1_mfail.timeseries.V_oc']

            plt.subplot(2, 2, 2)
            plt.plot(t0, V_oc0, 'b')
            plt.plot(t1, V_oc1, 'b')
            plt.plot(t1b, V_oc1b, 'r')
            plt.plot(t1m, V_oc1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('Open Circuit Voltage (V)')

            V_pack0 = prob['traj.phase0.timeseries.V_pack']
            V_pack1 = prob['traj.phase1.timeseries.V_pack']
            V_pack1b = prob['traj.phase1_bfail.timeseries.V_pack']
            V_pack1m = prob['traj.phase1_mfail.timeseries.V_pack']

            plt.subplot(2, 2, 3)
            plt.plot(t0, V_pack0, 'b')
            plt.plot(t1, V_pack1, 'b')
            plt.plot(t1b, V_pack1b, 'r')
            plt.plot(t1m, V_pack1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('Terminal Voltage (V)')

            I_Li0 = prob['traj.phase0.timeseries.I_Li']
            I_Li1 = prob['traj.phase1.timeseries.I_Li']
            I_Li1b = prob['traj.phase1_bfail.timeseries.I_Li']
            I_Li1m = prob['traj.phase1_mfail.timeseries.I_Li']

            plt.subplot(2, 2, 4)
            plt.plot(t0, I_Li0, 'b')
            plt.plot(t1, I_Li1, 'b')
            plt.plot(t1b, I_Li1b, 'r')
            plt.plot(t1m, I_Li1m, 'c')
            plt.xlabel('Time (hour)')
            plt.ylabel('Line Current (A)')

            plt.show()


if __name__ == '__main__':
    unittest.main()
