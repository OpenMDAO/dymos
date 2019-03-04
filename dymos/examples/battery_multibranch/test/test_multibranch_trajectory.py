"""
Integration test for a battery+motor example that demonstrates phase branching in trajectories.
"""
from __future__ import print_function, division, absolute_import

import os
import unittest

from openmdao.api import Problem, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase, Trajectory
from dymos.examples.battery_multibranch.battery_multibranch_ode import BatteryODE
from dymos.utils.lgl import lgl

optimizer = os.environ.get('DYMOS_DEFAULT_OPT', 'SLSQP')


class TestBatteryBranchingPhases(unittest.TestCase):

    def test_optimizer_defects(self):
        transcription = 'radau-ps'
        prob = Problem()

        if optimizer == 'SNOPT':
            opt = prob.driver = pyOptSparseDriver()
            opt.options['optimizer'] = optimizer
            opt.options['dynamic_simul_derivs'] = True

            opt.opt_settings['Major iterations limit'] = 1000
            opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
            opt.opt_settings['Major optimality tolerance'] = 1.0E-6
            opt.opt_settings["Linesearch tolerance"] = 0.10
            opt.opt_settings['iSumm'] = 6

        else:
            opt = prob.driver = ScipyOptimizeDriver()
            opt.options['dynamic_simul_derivs'] = True

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

        prob['traj.phase0.t_initial'] = 0
        prob['traj.phase0.t_duration'] = 1.0*3600

        prob['traj.phase1.t_initial'] = 1.0*3600
        prob['traj.phase1.t_duration'] = 1.0*3600

        prob['traj.phase1_bfail.t_initial'] = 1.0*3600
        prob['traj.phase1_bfail.t_duration'] = 1.0*3600

        prob['traj.phase1_mfail.t_initial'] = 1.0*3600
        prob['traj.phase1_mfail.t_duration'] = 1.0*3600

        # Fail one battery
        prob.model.traj.phases.phase1_bfail.rhs_all.battery.options['n_parallel'] = 2

        # Fail one motor
        prob.model.traj.phases.phase1_mfail.rhs_all.motors.options['n_parallel'] = 2

        prob.set_solver_print(level=0)
        prob.run_driver()

        soc0 = prob['traj.phase0.states:state_of_charge']
        soc1 = prob['traj.phase1.states:state_of_charge']
        soc1b = prob['traj.phase1_bfail.states:state_of_charge']
        soc1m = prob['traj.phase1_mfail.states:state_of_charge']

        # Final value for State of Chrage in each segment should be a good test.
        assert_rel_error(self, soc0[-1], 0.63464982, 1e-6)
        assert_rel_error(self, soc1[-1], 0.23794217, 1e-6)
        assert_rel_error(self, soc1b[-1], 0.0281523, 1e-6)
        assert_rel_error(self, soc1m[-1], 0.18625395, 1e-6)

    def test_solver_defects(self):
        transcription = 'radau-ps'
        prob = Problem()

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj', Trajectory())

        # First phase: normal operation.

        phase0 = Phase(transcription,
                       ode_class=BatteryODE,
                       num_segments=num_seg,
                       segment_ends=seg_ends,
                       transcription_order=5,
                       compressed=True)

        traj_p0 = traj.add_phase('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.set_state_options('state_of_charge', fix_initial=True, fix_final=False,
                                  solve_segments=True)

        # Second phase: normal operation.

        phase1 = Phase(transcription,
                       ode_class=BatteryODE,
                       num_segments=num_seg,
                       segment_ends=seg_ends,
                       transcription_order=5,
                       compressed=True)

        traj_p1 = traj.add_phase('phase1', phase1)

        traj_p1.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1.set_state_options('state_of_charge', fix_initial=False, fix_final=False,
                                  solve_segments=True)
        traj_p1.add_objective('time', loc='final')

        # Second phase, but with battery failure.

        phase1_bfail = Phase(transcription,
                             ode_class=BatteryODE,
                             num_segments=num_seg,
                             segment_ends=seg_ends,
                             transcription_order=5,
                             compressed=True)

        traj_p1_bfail = traj.add_phase('phase1_bfail', phase1_bfail)

        traj_p1_bfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_bfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False,
                                        solve_segments=True)

        # Second phase, but with motor failure.

        phase1_mfail = Phase(transcription,
                             ode_class=BatteryODE,
                             num_segments=num_seg,
                             segment_ends=seg_ends,
                             transcription_order=5,
                             compressed=True)

        traj_p1_mfail = traj.add_phase('phase1_mfail', phase1_mfail)

        traj_p1_mfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_mfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False,
                                        solve_segments=True)

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'],
                         connected=True)

        prob.setup()

        prob['traj.phase0.t_initial'] = 0
        prob['traj.phase0.t_duration'] = 1.0*3600

        prob['traj.phase1.t_initial'] = 1.0*3600
        prob['traj.phase1.t_duration'] = 1.0*3600

        prob['traj.phase1_bfail.t_initial'] = 1.0*3600
        prob['traj.phase1_bfail.t_duration'] = 1.0*3600

        prob['traj.phase1_mfail.t_initial'] = 1.0*3600
        prob['traj.phase1_mfail.t_duration'] = 1.0*3600

        prob['traj.phase0.states:state_of_charge'][:] = 1.0

        # Fail one battery
        prob.model.traj.phases.phase1_bfail.rhs_all.battery.options['n_parallel'] = 2

        # Fail one motor
        prob.model.traj.phases.phase1_mfail.rhs_all.motors.options['n_parallel'] = 2

        prob.set_solver_print(level=0)
        prob.run_model()

        soc0 = prob['traj.phase0.states:state_of_charge']
        soc1 = prob['traj.phase1.states:state_of_charge']
        soc1b = prob['traj.phase1_bfail.states:state_of_charge']
        soc1m = prob['traj.phase1_mfail.states:state_of_charge']

        # Final value for State of Charge in each segment should be a good test.
        assert_rel_error(self, soc0[-1], 0.63464982, 1e-6)
        assert_rel_error(self, soc1[-1], 0.23794217, 1e-6)
        assert_rel_error(self, soc1b[-1], 0.0281523, 1e-6)
        assert_rel_error(self, soc1m[-1], 0.18625395, 1e-6)

    def test_solver_defects_single_phase_reverse_propagation(self):
        transcription = 'radau-ps'
        prob = Problem()

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        # First phase: normal operation.

        phase0 = Phase(transcription,
                       ode_class=BatteryODE,
                       num_segments=num_seg,
                       segment_ends=seg_ends,
                       transcription_order=5,
                       compressed=True)

        traj_p0 = prob.model.add_subsystem('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.set_state_options('state_of_charge', fix_initial=False, fix_final=True,
                                  solve_segments=True)

        prob.setup()

        prob['phase0.t_initial'] = 0
        prob['phase0.t_duration'] = -1.0*3600
        prob['phase0.states:state_of_charge'][:] = 0.63464982

        prob.set_solver_print(level=0)
        prob.run_model()

        soc0 = prob['phase0.states:state_of_charge']
        assert_rel_error(self, soc0[-1], 1.0, 1e-6)

    def test_solver_defects_reverse_propagation(self):
        transcription = 'radau-ps'
        prob = Problem()

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj', Trajectory())

        # First phase: normal operation.

        phase0 = Phase(transcription,
                       ode_class=BatteryODE,
                       num_segments=num_seg,
                       segment_ends=seg_ends,
                       transcription_order=5,
                       compressed=True)

        traj_p0 = traj.add_phase('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.set_state_options('state_of_charge', fix_initial=True, fix_final=False,
                                  solve_segments=True)

        # Second phase: normal operation.

        phase1 = Phase(transcription,
                       ode_class=BatteryODE,
                       num_segments=num_seg,
                       segment_ends=seg_ends,
                       transcription_order=5,
                       compressed=True)

        traj_p1 = traj.add_phase('phase1', phase1)

        traj_p1.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1.set_state_options('state_of_charge', fix_initial=False, fix_final=False,
                                  solve_segments=True)
        traj_p1.add_objective('time', loc='final')

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'],
                         connected=True)

        prob.setup()

        prob['traj.phase0.t_initial'] = 0
        prob['traj.phase0.t_duration'] = -1.0*3600
        prob['traj.phase0.states:state_of_charge'][:] = 0.23794217

        prob['traj.phase1.t_initial'] = 0
        prob['traj.phase1.t_duration'] = -1.0*3600

        prob.set_solver_print(level=0)
        prob.run_model()

        soc1 = prob['traj.phase1.states:state_of_charge']
        assert_rel_error(self, soc1[-1], 1.0, 1e-6)

    def test_optimizer_segments_direct_connections(self):
        transcription = 'radau-ps'
        prob = Problem()

        if optimizer == 'SNOPT':
            opt = prob.driver = pyOptSparseDriver()
            opt.options['optimizer'] = optimizer
            opt.options['dynamic_simul_derivs'] = True

            opt.opt_settings['Major iterations limit'] = 1000
            opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
            opt.opt_settings['Major optimality tolerance'] = 1.0E-6
            opt.opt_settings["Linesearch tolerance"] = 0.10
            opt.opt_settings['iSumm'] = 6

        else:
            opt = prob.driver = ScipyOptimizeDriver()
            opt.options['dynamic_simul_derivs'] = True

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj', Trajectory())

        # First phase: normal operation.

        phase0 = Phase(transcription,
                       ode_class=BatteryODE,
                       num_segments=num_seg,
                       segment_ends=seg_ends,
                       transcription_order=5,
                       compressed=True)

        traj_p0 = traj.add_phase('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.set_state_options('state_of_charge', fix_initial=True, fix_final=False)

        # Second phase: normal operation.

        phase1 = Phase(transcription,
                       ode_class=BatteryODE,
                       num_segments=num_seg,
                       segment_ends=seg_ends,
                       transcription_order=5,
                       compressed=True)

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
                             compressed=True)

        traj_p1_bfail = traj.add_phase('phase1_bfail', phase1_bfail)

        traj_p1_bfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_bfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False)

        # Second phase, but with motor failure.

        phase1_mfail = Phase(transcription,
                             ode_class=BatteryODE,
                             num_segments=num_seg,
                             segment_ends=seg_ends,
                             transcription_order=5,
                             compressed=True)

        traj_p1_mfail = traj.add_phase('phase1_mfail', phase1_mfail)

        traj_p1_mfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_mfail.set_state_options('state_of_charge', fix_initial=False, fix_final=False)

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'],
                         connected=True)

        prob.model.options['assembled_jac_type'] = 'csc'
        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.setup()

        prob['traj.phase0.t_initial'] = 0
        prob['traj.phase0.t_duration'] = 1.0*3600

        prob['traj.phase1.t_initial'] = 1.0*3600
        prob['traj.phase1.t_duration'] = 1.0*3600

        prob['traj.phase1_bfail.t_initial'] = 1.0*3600
        prob['traj.phase1_bfail.t_duration'] = 1.0*3600

        prob['traj.phase1_mfail.t_initial'] = 1.0*3600
        prob['traj.phase1_mfail.t_duration'] = 1.0*3600

        # Fail one battery
        prob.model.traj.phases.phase1_bfail.rhs_all.battery.options['n_parallel'] = 2

        # Fail one motor
        prob.model.traj.phases.phase1_mfail.rhs_all.motors.options['n_parallel'] = 2

        prob.set_solver_print(level=0)
        prob.run_driver()

        soc0 = prob['traj.phase0.states:state_of_charge']
        soc1 = prob['traj.phase1.states:state_of_charge']
        soc1b = prob['traj.phase1_bfail.states:state_of_charge']
        soc1m = prob['traj.phase1_mfail.states:state_of_charge']

        # Final value for State of Chrage in each segment should be a good test.
        assert_rel_error(self, soc0[-1], 0.63464982, 1e-6)
        assert_rel_error(self, soc1[-1], 0.23794217, 1e-6)
        assert_rel_error(self, soc1b[-1], 0.0281523, 1e-6)
        assert_rel_error(self, soc1m[-1], 0.18625395, 1e-6)


if __name__ == '__main__':
    unittest.main()
