"""
Integration test for a battery+motor example that demonstrates phase branching in trajectories.
"""
import unittest

import matplotlib
matplotlib.use('Agg')

from openmdao.utils.testing_utils import use_tempdirs
from dymos.utils.doc_utils import save_for_docs


@use_tempdirs
class TestBatteryBranchingPhasesForDocs(unittest.TestCase):

    @save_for_docs
    def test_basic(self):
        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.battery_multibranch.battery_multibranch_ode import BatteryODE
        from dymos.utils.lgl import lgl

        prob = om.Problem()

        opt = prob.driver = om.ScipyOptimizeDriver()
        opt.declare_coloring()
        opt.options['optimizer'] = 'SLSQP'

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj', dm.Trajectory())

        # First phase: normal operation.
        transcription = dm.Radau(num_segments=num_seg, order=5, segment_ends=seg_ends, compressed=False)
        phase0 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p0 = traj.add_phase('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.add_state('state_of_charge', fix_initial=True, fix_final=False,
                          targets=['SOC'], rate_source='dXdt:SOC')

        # Second phase: normal operation.

        phase1 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p1 = traj.add_phase('phase1', phase1)

        traj_p1.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1.add_state('state_of_charge', fix_initial=False, fix_final=False,
                          targets=['SOC'], rate_source='dXdt:SOC')
        traj_p1.add_objective('time', loc='final')

        # Second phase, but with battery failure.

        phase1_bfail = dm.Phase(ode_class=BatteryODE, ode_init_kwargs={'num_battery': 2},
                                transcription=transcription)
        traj_p1_bfail = traj.add_phase('phase1_bfail', phase1_bfail)

        traj_p1_bfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_bfail.add_state('state_of_charge', fix_initial=False, fix_final=False,
                                targets=['SOC'], rate_source='dXdt:SOC')

        # Second phase, but with motor failure.

        phase1_mfail = dm.Phase(ode_class=BatteryODE, ode_init_kwargs={'num_motor': 2},
                                transcription=transcription)
        traj_p1_mfail = traj.add_phase('phase1_mfail', phase1_mfail)

        traj_p1_mfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_mfail.add_state('state_of_charge', fix_initial=False, fix_final=False,
                                targets=['SOC'], rate_source='dXdt:SOC')

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'])
        traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'])
        traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'])

        prob.model.options['assembled_jac_type'] = 'csc'
        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup()

        prob['traj.phase0.t_initial'] = 0
        prob['traj.phase0.t_duration'] = 1.0*3600

        prob['traj.phase1.t_initial'] = 1.0*3600
        prob['traj.phase1.t_duration'] = 1.0*3600

        prob['traj.phase1_bfail.t_initial'] = 1.0*3600
        prob['traj.phase1_bfail.t_duration'] = 1.0*3600

        prob['traj.phase1_mfail.t_initial'] = 1.0*3600
        prob['traj.phase1_mfail.t_duration'] = 1.0*3600

        prob.set_solver_print(level=0)
        dm.run_problem(prob)

        soc0 = prob['traj.phase0.states:state_of_charge']
        soc1 = prob['traj.phase1.states:state_of_charge']
        soc1b = prob['traj.phase1_bfail.states:state_of_charge']
        soc1m = prob['traj.phase1_mfail.states:state_of_charge']

        # Final value for State of Chrage in each segment should be a good test.
        print('State of Charge after 1 hour')
        assert_near_equal(soc0[-1], 0.63464982, 1e-6)
        print('State of Charge after 2 hours')
        assert_near_equal(soc1[-1], 0.23794217, 1e-6)
        print('State of Charge after 2 hours, battery fails at 1 hour')
        assert_near_equal(soc1b[-1], 0.0281523, 1e-6)
        print('State of Charge after 2 hours, motor fails at 1 hour')
        assert_near_equal(soc1m[-1], 0.18625395, 1e-6)

        # Plot Results
        t0 = prob['traj.phases.phase0.time.time']/3600
        t1 = prob['traj.phases.phase1.time.time']/3600
        t1b = prob['traj.phases.phase1_bfail.time.time']/3600
        t1m = prob['traj.phases.phase1_mfail.time.time']/3600

        plt.subplot(2, 1, 1)
        plt.plot(t0, soc0, 'b')
        plt.plot(t1, soc1, 'b')
        plt.plot(t1b, soc1b, 'r')
        plt.plot(t1m, soc1m, 'c')
        plt.xlabel('Time (hour)')
        plt.ylabel('State of Charge (percent)')

        I_Li0 = prob['traj.phases.phase0.rhs_all.pwr_balance.I_Li']
        I_Li1 = prob['traj.phases.phase1.rhs_all.pwr_balance.I_Li']
        I_Li1b = prob['traj.phases.phase1_bfail.rhs_all.pwr_balance.I_Li']
        I_Li1m = prob['traj.phases.phase1_mfail.rhs_all.pwr_balance.I_Li']

        plt.subplot(2, 1, 2)
        plt.plot(t0, I_Li0, 'b')
        plt.plot(t1, I_Li1, 'b')
        plt.plot(t1b, I_Li1b, 'r')
        plt.plot(t1m, I_Li1m, 'c')
        plt.xlabel('Time (hour)')
        plt.ylabel('Line Current (A)')

        plt.legend(['Phase 1', 'Phase 2', 'Phase 2 Battery Fail', 'Phase 2 Motor Fail'], loc=2)

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
