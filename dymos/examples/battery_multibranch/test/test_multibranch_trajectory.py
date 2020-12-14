"""
Integration test for a battery+motor example that demonstrates phase branching in trajectories.
"""
import os
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.battery_multibranch.battery_multibranch_ode import BatteryODE
from dymos.utils.lgl import lgl

optimizer = os.environ.get('DYMOS_DEFAULT_OPT', 'SLSQP')


@use_tempdirs
class TestBatteryBranchingPhases(unittest.TestCase):
    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_optimizer_defects(self):

        prob = om.Problem()

        opt = prob.driver = om.pyOptSparseDriver()
        opt.declare_coloring()

        if optimizer == 'SNOPT':
            opt.options['optimizer'] = optimizer
            opt.opt_settings['Major iterations limit'] = 1000
            opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
            opt.opt_settings['Major optimality tolerance'] = 1.0E-6
            opt.opt_settings["Linesearch tolerance"] = 0.10
            opt.opt_settings['iSumm'] = 6

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj',  dm.Trajectory())

        # First phase: normal operation.
        transcription = dm.Radau(num_segments=5, order=5, segment_ends=seg_ends, compressed=False)
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
        assert_near_equal(soc0[-1], 0.63464982, 1e-6)
        assert_near_equal(soc1[-1], 0.23794217, 1e-6)
        assert_near_equal(soc1b[-1], 0.0281523, 1e-6)
        assert_near_equal(soc1m[-1], 0.18625395, 1e-6)

    def test_solver_defects(self):
        prob = om.Problem()

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj',  dm.Trajectory())

        # First phase: normal operation.

        transcription = dm.Radau(num_segments=5, order=5, segment_ends=seg_ends, compressed=True)
        phase0 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p0 = traj.add_phase('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.add_state('state_of_charge', fix_initial=True, fix_final=False,
                          solve_segments='forward',
                          targets=['SOC'], rate_source='dXdt:SOC')

        # Second phase: normal operation.
        phase1 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p1 = traj.add_phase('phase1', phase1)

        traj_p1.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1.add_state('state_of_charge', fix_initial=False, fix_final=False,
                          solve_segments='forward',
                          targets=['SOC'], rate_source='dXdt:SOC')
        traj_p1.add_objective('time', loc='final')

        # Second phase, but with battery failure.
        phase1_bfail = dm.Phase(ode_class=BatteryODE, ode_init_kwargs={'num_battery': 2},
                                transcription=transcription)
        traj_p1_bfail = traj.add_phase('phase1_bfail', phase1_bfail)

        traj_p1_bfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_bfail.add_state('state_of_charge', fix_initial=False, fix_final=False,
                                solve_segments='forward',
                                targets=['SOC'], rate_source='dXdt:SOC')

        # Second phase, but with motor failure.
        phase1_mfail = dm.Phase(ode_class=BatteryODE, ode_init_kwargs={'num_motor': 2},
                                transcription=transcription)
        traj_p1_mfail = traj.add_phase('phase1_mfail', phase1_mfail)

        traj_p1_mfail.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1_mfail.add_state('state_of_charge', fix_initial=False, fix_final=False,
                                solve_segments='forward',
                                targets=['SOC'], rate_source='dXdt:SOC')

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

        prob.set_solver_print(level=0)
        prob.run_model()

        soc0 = prob['traj.phase0.states:state_of_charge']
        soc1 = prob['traj.phase1.states:state_of_charge']
        soc1b = prob['traj.phase1_bfail.states:state_of_charge']
        soc1m = prob['traj.phase1_mfail.states:state_of_charge']

        # Final value for State of Charge in each segment should be a good test.
        assert_near_equal(soc0[-1], 0.63464982, 1e-6)
        assert_near_equal(soc1[-1], 0.23794217, 1e-6)
        assert_near_equal(soc1b[-1], 0.0281523, 1e-6)
        assert_near_equal(soc1m[-1], 0.18625395, 1e-6)

    def test_solver_defects_single_phase_reverse_propagation(self):
        prob = om.Problem()

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        # First phase: normal operation.

        transcription = dm.Radau(num_segments=5, order=5, segment_ends=seg_ends, compressed=True)
        phase0 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p0 = prob.model.add_subsystem('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.add_state('state_of_charge', fix_initial=True, fix_final=False,
                          solve_segments='forward',
                          targets=['SOC'], rate_source='dXdt:SOC')

        prob.setup()

        prob['phase0.t_initial'] = 0
        prob['phase0.t_duration'] = -1.0*3600
        prob['phase0.states:state_of_charge'][:] = 0.63464982

        prob.set_solver_print(level=0)
        prob.run_model()

        soc0 = prob['phase0.states:state_of_charge']
        assert_near_equal(soc0[-1], 1.0, 1e-6)

    def test_solver_defects_reverse_propagation(self):
        prob = om.Problem()

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj',  dm.Trajectory())

        # First phase: normal operation.
        transcription = dm.Radau(num_segments=5, order=5, segment_ends=seg_ends, compressed=True)
        phase0 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p0 = traj.add_phase('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.add_state('state_of_charge', fix_initial=True, fix_final=False,
                          solve_segments='forward',
                          targets=['SOC'], rate_source='dXdt:SOC')

        # Second phase: normal operation.
        phase1 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p1 = traj.add_phase('phase1', phase1)

        traj_p1.set_time_options(fix_initial=False, fix_duration=True)
        traj_p1.add_state('state_of_charge', fix_initial=False, fix_final=False,
                          solve_segments='forward',
                          targets=['SOC'], rate_source='dXdt:SOC')
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
        assert_near_equal(soc1[-1], 1.0, 1e-6)

    def test_optimizer_segments_direct_connections(self):
        prob = om.Problem()

        if optimizer == 'SNOPT':
            opt = prob.driver = om.pyOptSparseDriver()
            opt.options['optimizer'] = optimizer
            opt.declare_coloring()

            opt.opt_settings['Major iterations limit'] = 1000
            opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
            opt.opt_settings['Major optimality tolerance'] = 1.0E-6
            opt.opt_settings["Linesearch tolerance"] = 0.10
            opt.opt_settings["Verify level"] = 3
            opt.opt_settings['iSumm'] = 6

        else:
            opt = prob.driver = om.ScipyOptimizeDriver()
            opt.declare_coloring()

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj',  dm.Trajectory())

        # First phase: normal operation.
        transcription = dm.Radau(num_segments=5, order=5, segment_ends=seg_ends, compressed=True)
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

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'],
                         connected=True)

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup(force_alloc_complex=True)

        prob['traj.phase0.t_initial'] = 0
        prob['traj.phase0.t_duration'] = 1.0*3600

        prob['traj.phase1.t_initial'] = 1.0*3600
        prob['traj.phase1.t_duration'] = 1.0*3600

        prob['traj.phase1_bfail.t_initial'] = 1.0*3600
        prob['traj.phase1_bfail.t_duration'] = 1.0*3600

        prob['traj.phase1_mfail.t_initial'] = 1.0*3600
        prob['traj.phase1_mfail.t_duration'] = 1.0*3600

        prob.run_model()

        dm.run_problem(prob)

        soc0 = prob['traj.phase0.states:state_of_charge']
        soc1 = prob['traj.phase1.states:state_of_charge']
        soc1b = prob['traj.phase1_bfail.states:state_of_charge']
        soc1m = prob['traj.phase1_mfail.states:state_of_charge']

        # Final value for State of Chrage in each segment should be a good test.
        assert_near_equal(soc0[-1], 0.63464982, 1e-6)
        assert_near_equal(soc1[-1], 0.23794217, 1e-6)
        assert_near_equal(soc1b[-1], 0.0281523, 1e-6)
        assert_near_equal(soc1m[-1], 0.18625395, 1e-6)


@use_tempdirs
class TestBatteryBranchingPhasesRungeKutta(unittest.TestCase):

    def test_constraint_linkages_rk(self):
        prob = om.Problem()

        if optimizer == 'SNOPT':
            opt = prob.driver = om.pyOptSparseDriver()
            opt.options['optimizer'] = optimizer
            opt.declare_coloring()

            opt.opt_settings['Major iterations limit'] = 1000
            opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
            opt.opt_settings['Major optimality tolerance'] = 1.0E-6
            opt.opt_settings["Linesearch tolerance"] = 0.10
            opt.opt_settings['iSumm'] = 6

        else:
            opt = prob.driver = om.ScipyOptimizeDriver()
            opt.declare_coloring()

        num_seg = 20
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj',  dm.Trajectory())

        # First phase: normal operation.
        transcription = dm.RungeKutta(num_segments=num_seg)
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

        # Final value for State of Charge in each segment should be a good test.
        assert_near_equal(soc0[-1], 0.63464982, 1e-4)
        assert_near_equal(soc1[-1], 0.23794217, 1e-4)
        assert_near_equal(soc1b[-1], 0.0281523, 1e-4)
        assert_near_equal(soc1m[-1], 0.18625395, 1e-4)

    def test_single_phase_reverse_propagation_rk(self):
        prob = om.Problem()

        num_seg = 10
        seg_ends, _ = lgl(num_seg + 1)

        # First phase: normal operation.
        transcription = dm.RungeKutta(num_segments=num_seg)
        phase0 = dm.Phase(ode_class=BatteryODE, transcription=transcription)
        traj_p0 = prob.model.add_subsystem('phase0', phase0)

        traj_p0.set_time_options(fix_initial=True, fix_duration=True)
        traj_p0.add_state('state_of_charge', fix_initial=True, fix_final=False,
                          targets=['SOC'], rate_source='dXdt:SOC')

        prob.setup()

        prob['phase0.t_initial'] = 0
        prob['phase0.t_duration'] = -1.0*3600
        prob['phase0.states:state_of_charge'][:] = 0.63464982

        prob.set_solver_print(level=0)
        prob.run_model()

        soc0 = prob['phase0.states:state_of_charge']
        assert_near_equal(soc0[-1], 1.0, 1e-6)

    def test_connected_linkages_rk(self):
        prob = om.Problem()

        if optimizer == 'SNOPT':
            opt = prob.driver = om.pyOptSparseDriver()
            opt.options['optimizer'] = optimizer
            opt.declare_coloring()

            opt.opt_settings['Major iterations limit'] = 1000
            opt.opt_settings['Major feasibility tolerance'] = 1.0E-6
            opt.opt_settings['Major optimality tolerance'] = 1.0E-6
            opt.opt_settings["Linesearch tolerance"] = 0.10
            opt.opt_settings['iSumm'] = 6

        else:
            opt = prob.driver = om.ScipyOptimizeDriver()
            opt.declare_coloring()

        num_seg = 20
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj',  dm.Trajectory())

        # First phase: normal operation.

        transcription = dm.RungeKutta(num_segments=num_seg)
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

        traj.link_phases(phases=['phase0', 'phase1'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_bfail'], vars=['state_of_charge', 'time'],
                         connected=True)
        traj.link_phases(phases=['phase0', 'phase1_mfail'], vars=['state_of_charge', 'time'],
                         connected=True)

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
        assert_near_equal(soc0[-1], 0.63464982, 5e-5)
        assert_near_equal(soc1[-1], 0.23794217, 5e-5)
        assert_near_equal(soc1b[-1], 0.0281523, 5e-5)
        assert_near_equal(soc1m[-1], 0.18625395, 5e-5)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
