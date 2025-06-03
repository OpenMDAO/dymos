import unittest
import pathlib

import numpy as np

try:
    import matplotlib
except ImportError:
    matplotlib = None

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.examples.battery_multibranch.battery_multibranch_ode import BatteryODE
from dymos.utils.lgl import lgl
from dymos.utils.testing_utils import _get_reports_dir
from dymos.visualization.timeseries_plots import timeseries_plots


@use_tempdirs
@require_pyoptsparse(optimizer='SLSQP')
class TestTimeSeriesPlotsBasics(unittest.TestCase):

    def setUp(self):
        optimizer = 'SLSQP'
        num_segments = 8
        transcription_order = 3
        compressed = True

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring()

        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
        phase.timeseries_options['include_control_rates'] = True
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

        phase.add_timeseries('timeseries2',
                             transcription=dm.Radau(num_segments=num_segments * 5,
                                                    order=transcription_order,
                                                    compressed=compressed),
                             subset='control_input')

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=['unconnected_inputs'])

        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.setup()

        self.p = p

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_timeseries_plots(self):
        with dm.options.temporary(plots='matplotlib'):
            dm.run_problem(self.p, make_plots=False)

            sol_db = self.p.get_outputs_dir() / 'dymos_solution.db'

            timeseries_plots(sol_db, problem=self.p)
            plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath('plots')

            self.assertTrue(plot_dir.joinpath('x.png').exists())
            self.assertTrue(plot_dir.joinpath('y.png').exists())
            self.assertTrue(plot_dir.joinpath('v.png').exists())
            self.assertTrue(plot_dir.joinpath('theta.png').exists())
            self.assertTrue(plot_dir.joinpath('theta_rate.png').exists())
            self.assertTrue(plot_dir.joinpath('theta_rate2.png').exists())

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_timeseries_plots_solution_only_set_solution_record_file(self):
        temp = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        # records to the default file 'dymos_simulation.db'
        dm.run_problem(self.p, make_plots=False, solution_record_file='solution_record_file.db')

        sol_db = self.p.get_outputs_dir() / 'solution_record_file.db'

        timeseries_plots(sol_db, problem=self.p)
        plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath('plots')

        self.assertTrue(plot_dir.joinpath('x.png').exists())
        self.assertTrue(plot_dir.joinpath('y.png').exists())
        self.assertTrue(plot_dir.joinpath('v.png').exists())
        self.assertTrue(plot_dir.joinpath('theta.png').exists())
        self.assertTrue(plot_dir.joinpath('theta_rate.png').exists())
        self.assertTrue(plot_dir.joinpath('theta_rate2.png').exists())

        dm.options['plots'] = temp

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_timeseries_plots_solution_and_simulation(self):
        temp = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        dm.run_problem(self.p, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        sol_db = self.p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = self.p.model.traj0.sim_prob.get_outputs_dir() / 'simulation_record_file.db'

        timeseries_plots(sol_db,
                         simulation_record_file=sim_db,
                         problem=self.p)

        plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath("plots").resolve()
        self.assertTrue(plot_dir.joinpath('x.png').exists())
        self.assertTrue(plot_dir.joinpath('y.png').exists())
        self.assertTrue(plot_dir.joinpath('v.png').exists())
        self.assertTrue(plot_dir.joinpath('theta.png').exists())
        self.assertTrue(plot_dir.joinpath('theta_rate.png').exists())
        self.assertTrue(plot_dir.joinpath('theta_rate2.png').exists())

        dm.options['plots'] = temp

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_timeseries_plots_set_plot_dir(self):

        temp = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        dm.run_problem(self.p, make_plots=False)

        sol_db = self.p.get_outputs_dir() / 'dymos_solution.db'

        plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath("test_plot_dir").resolve()
        timeseries_plots(sol_db, plot_dir=plot_dir)

        self.assertTrue(plot_dir.joinpath('x.png').exists())
        self.assertTrue(plot_dir.joinpath('y.png').exists())
        self.assertTrue(plot_dir.joinpath('v.png').exists())
        self.assertTrue(plot_dir.joinpath('theta.png').exists())
        self.assertTrue(plot_dir.joinpath('theta_rate.png').exists())
        self.assertTrue(plot_dir.joinpath('theta_rate2.png').exists())

        dm.options['plots'] = temp


@use_tempdirs
class TestTimeSeriesPlotsMultiPhase(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_trajectory_linked_phases_make_plot(self):
        temp = dm.options['plots']

        dm.options['plots'] = 'matplotlib'

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.declare_coloring()

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4,
                                                                                order=3))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10,
                                                                                order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10,
                               units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', targets=['u1'], opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3,
                                                                                order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10,
                               units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        for phase_name, phase in self.traj._phases.items():
            phase.timeseries_options['use_prefix'] = True
            phase.timeseries_options['include_state_rates'] = True
            phase.timeseries_options['include_t_phase'] = True
            phase.timeseries_options['include_control_rates'] = True

        p.setup(check=True)

        # Set Initial Guesses
        burn1.set_time_val(initial=0.0, duration=2.25)
        burn1.set_state_val('r', [1, 1.5])
        burn1.set_state_val('theta', [0, 1.7])
        burn1.set_state_val('vr', [0, 0])
        burn1.set_state_val('vt', [1, 1])
        burn1.set_state_val('accel', [0.1, 0])
        burn1.set_state_val('deltav', [0, 0.1])
        burn1.set_control_val('u1', [-3.5, 13.0])
        burn1.set_parameter_val('c', 1.5)

        coast.set_time_val(initial=2.25, duration=3.0)
        coast.set_state_val('r', [1.3, 1.5])
        coast.set_state_val('theta', [2.1767, 1.7])
        coast.set_state_val('vr', [0.3285, 0.0])
        coast.set_state_val('vt', [0.97, 1])
        coast.set_state_val('accel', [0.0, 0.0])
        coast.set_parameter_val('c', 1.5)

        burn2.set_time_val(initial=5.25, duration=1.75)
        burn2.set_state_val('r', [1, 3])
        burn2.set_state_val('theta', [0, 4])
        burn2.set_state_val('vr', [0, 0])
        burn2.set_state_val('vt', [1, np.sqrt(1./3.)])
        burn2.set_state_val('accel', [0.1, 0])
        burn2.set_state_val('deltav', [0.1, 0.2])
        burn2.set_control_val('u1', [1.0, 1.0])
        burn2.set_parameter_val('c', 1.5)

        dm.run_problem(p, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        sol_db = self.p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = self.p.model.sim_prob.get_outputs_dir() / 'simulation_record_file.db'

        timeseries_plots(sol_db,
                         simulation_record_file=sim_db,
                         problem=p)
        plot_dir = pathlib.Path(_get_reports_dir(p)).joinpath("plots")

        for varname in ['time_phase', 'states:r', 'state_rates:r', 'states:theta',
                        'state_rates:theta', 'states:vr', 'state_rates:vr', 'states:vt',
                        'state_rates:vt', 'states:accel',
                        'state_rates:accel', 'states:deltav', 'state_rates:deltav',
                        'controls:u1', 'control_rates:u1_rate', 'control_rates:u1_rate2']:
            plotfile = plot_dir.joinpath(varname.replace(":", "_") + '.png')
            self.assertTrue(plotfile.exists(), msg=f'{plotfile} does not exist!')

        dm.options['plots'] = temp

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_overlapping_phases_make_plot(self):

        _temp = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        prob = om.Problem()

        opt = prob.driver = om.ScipyOptimizeDriver()
        opt.declare_coloring()
        opt.options['optimizer'] = 'SLSQP'

        num_seg = 5
        seg_ends, _ = lgl(num_seg + 1)

        traj = prob.model.add_subsystem('traj', dm.Trajectory())

        # First phase: normal operation.
        transcription = dm.Radau(num_segments=num_seg, order=5, segment_ends=seg_ends,
                                 compressed=False)
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

        for phase_name, phase in traj._phases.items():
            phase.timeseries_options['include_t_phase'] = True
            phase.timeseries_options['include_state_rates'] = True
            phase.timeseries_options['include_control_rates'] = True

        prob.setup()

        phase0.set_time_val(initial=0.0, duration=3600)
        phase1.set_time_val(initial=3600.0, duration=3600.0)
        phase1_bfail.set_time_val(initial=3600.0, duration=3600.0)
        phase1_mfail.set_time_val(initial=3600.0, duration=3600.0)
        phase0.set_state_val('state_of_charge', 1.0)

        prob.set_solver_print(level=0)
        dm.run_problem(prob, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        sol_db = prob.get_outputs_dir() / 'dymos_solution.db'
        sim_db = prob.model.traj.sim_prob.get_outputs_dir() / 'simulation_record_file.db'

        plot_dir = pathlib.Path(_get_reports_dir(prob)).joinpath("plots")

        timeseries_plots(sol_db,
                         simulation_record_file=sim_db,
                         problem=prob)

        self.assertTrue(plot_dir.joinpath('time_phase.png').exists())
        self.assertTrue(plot_dir.joinpath('state_of_charge.png').exists())
        self.assertTrue(plot_dir.joinpath('state_of_charge.png').exists())

        dm.options['plots'] = _temp

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_trajectory_linked_phases_make_plot_missing_data(self):
        """
        Test that plots are still generated even if the phases don't share the exact same
        variables in the timeseries.
        """
        temp = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.declare_coloring()

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4,
                                                                                order=3))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU', include_timeseries=True)

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10,
                                                                                order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10,
                               units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg', include_timeseries=True)
        coast.add_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU', include_timeseries=True)

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3,
                                                                                order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10,
                               units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU', include_timeseries=True)

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        for phase_name, phase in self.traj._phases.items():
            phase.timeseries_options['include_t_phase'] = True
            phase.timeseries_options['include_state_rates'] = True
            phase.timeseries_options['include_control_rates'] = True

        p.setup(check=True)

        # Set Initial Guesses
        burn1.set_time_val(initial=0.0, duration=2.25)
        burn1.set_state_val('r', [1, 1.5])
        burn1.set_state_val('theta', [0, 1.7])
        burn1.set_state_val('vr', [0, 0])
        burn1.set_state_val('vt', [1, 1])
        burn1.set_state_val('accel', [0.1, 0])
        burn1.set_state_val('deltav', [0, 0.1])
        burn1.set_control_val('u1', [-3.5, 13.0])
        burn1.set_parameter_val('c', 1.5)

        coast.set_time_val(initial=2.25, duration=3.0)
        coast.set_state_val('r', [1.3, 1.5])
        coast.set_state_val('theta', [2.1767, 1.7])
        coast.set_state_val('vr', [0.3285, 0.0])
        coast.set_state_val('vt', [0.97, 1])
        coast.set_state_val('accel', [0.0, 0.0])
        coast.set_parameter_val('c', 1.5)

        burn2.set_time_val(initial=5.25, duration=1.75)
        burn2.set_state_val('r', [1, 3])
        burn2.set_state_val('theta', [0, 4])
        burn2.set_state_val('vr', [0, 0])
        burn2.set_state_val('vt', [1, np.sqrt(1./3.)])
        burn2.set_state_val('accel', [0.1, 0])
        burn2.set_state_val('deltav', [0.1, 0.2])
        burn2.set_control_val('u1', [1.0, 1.0])
        burn2.set_parameter_val('c', 1.5)

        dm.run_problem(p, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.sim_prob.get_outputs_dir() / 'simulation_record_file.db'

        timeseries_plots(sol_db, sim_db, problem=p)
        plot_dir = pathlib.Path(_get_reports_dir(p)).joinpath("plots")

        for varname in ['time_phase', 'r', 'r_dot', 'theta',
                        'theta_dot', 'vr', 'vr_dot', 'vt',
                        'vt_dot', 'accel',
                        'at_dot', 'deltav', 'deltav_dot',
                        'u1', 'u1_rate', 'u1_rate2']:
            self.assertTrue(plot_dir.joinpath(varname + '.png').exists(),
                            msg=varname + '.png does not exist!')

        dm.options['plots'] = temp


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
