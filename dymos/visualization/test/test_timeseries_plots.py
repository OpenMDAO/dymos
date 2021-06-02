import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.examples.battery_multibranch.battery_multibranch_ode import BatteryODE
from dymos.utils.lgl import lgl
from dymos.visualization.timeseries_plots import timeseries_plots
from dymos.utils.testing_utils import require_pyoptsparse


@use_tempdirs
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

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj0.phase0.parameters:g'] = 9.80665

        p.setup()

        self.p = p

    def test_brachistochrone_timeseries_plots(self):
        dm.run_problem(self.p, make_plots=False)

        timeseries_plots('dymos_solution.db')
        self.assertTrue(os.path.exists('plots/states_x.png'))
        self.assertTrue(os.path.exists('plots/states_y.png'))
        self.assertTrue(os.path.exists('plots/states_v.png'))
        self.assertTrue(os.path.exists('plots/controls_theta.png'))
        self.assertTrue(os.path.exists('plots/control_rates_theta_rate.png'))
        self.assertTrue(os.path.exists('plots/control_rates_theta_rate2.png'))

    def test_brachistochrone_timeseries_plots_solution_only_set_solution_record_file(self):
        # records to the default file 'dymos_simulation.db'
        dm.run_problem(self.p, make_plots=False, solution_record_file='solution_record_file.db')

        timeseries_plots('solution_record_file.db')
        self.assertTrue(os.path.exists('plots/states_x.png'))
        self.assertTrue(os.path.exists('plots/states_y.png'))
        self.assertTrue(os.path.exists('plots/states_v.png'))
        self.assertTrue(os.path.exists('plots/controls_theta.png'))
        self.assertTrue(os.path.exists('plots/control_rates_theta_rate.png'))
        self.assertTrue(os.path.exists('plots/control_rates_theta_rate2.png'))

    def test_brachistochrone_timeseries_plots_solution_and_simulation(self):
        dm.run_problem(self.p, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        timeseries_plots('dymos_solution.db', simulation_record_file='simulation_record_file.db')

    def test_brachistochrone_timeseries_plots_set_plot_dir(self):
        dm.run_problem(self.p, make_plots=False)

        plot_dir = "test_plot_dir"
        timeseries_plots('dymos_solution.db', plot_dir=plot_dir)

        self.assertTrue(os.path.exists('test_plot_dir/states_x.png'))
        self.assertTrue(os.path.exists('test_plot_dir/states_y.png'))
        self.assertTrue(os.path.exists('test_plot_dir/states_v.png'))
        self.assertTrue(os.path.exists('test_plot_dir/controls_theta.png'))
        self.assertTrue(os.path.exists('test_plot_dir/control_rates_theta_rate.png'))
        self.assertTrue(os.path.exists('test_plot_dir/control_rates_theta_rate2.png'))


@use_tempdirs
class TestTimeSeriesPlotsMultiPhase(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    def test_trajectory_linked_phases_make_plot(self):

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
        burn1.add_control('u1',  targets=['u1'], rate_continuity=True, rate2_continuity=True,
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

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        dm.run_problem(p, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        timeseries_plots('dymos_solution.db', simulation_record_file='simulation_record_file.db')

        for varname in ['time_phase', 'states:r', 'state_rates:r', 'states:theta',
                        'state_rates:theta', 'states:vr', 'state_rates:vr', 'states:vt',
                        'state_rates:vt', 'states:accel',
                        'state_rates:accel', 'states:deltav', 'state_rates:deltav',
                        'controls:u1', 'control_rates:u1_rate', 'control_rates:u1_rate2',
                        'parameters:c']:
            self.assertTrue(os.path.exists(f'plots/{varname.replace(":","_")}.png'))

    def test_overlapping_phases_make_plot(self):

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
        dm.run_problem(prob, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        timeseries_plots('dymos_solution.db', simulation_record_file='simulation_record_file.db')

        self.assertTrue(os.path.exists('plots/time_phase.png'))
        self.assertTrue(os.path.exists('plots/states_state_of_charge.png'))
        self.assertTrue(os.path.exists('plots/state_rates_state_of_charge.png'))

    @require_pyoptsparse(optimizer='IPOPT')
    def test_trajectory_linked_phases_make_plot_missing_data(self):
        """
        Test that plots are still generated even if the phases don't share the exact same
        variables in the timeseries.
        """

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
        burn1.add_control('u1',  targets=['u1'], rate_continuity=True, rate2_continuity=True,
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
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg')
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

        p.setup(check=True)

        # Set Initial Guesses
        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        dm.run_problem(p, simulate=True, make_plots=False,
                       simulation_record_file='simulation_record_file.db')

        timeseries_plots('dymos_solution.db', simulation_record_file='simulation_record_file.db')

        for varname in ['time_phase', 'states:r', 'state_rates:r', 'states:theta',
                        'state_rates:theta', 'states:vr', 'state_rates:vr', 'states:vt',
                        'state_rates:vt', 'states:accel',
                        'state_rates:accel', 'states:deltav', 'state_rates:deltav',
                        'controls:u1', 'control_rates:u1_rate', 'control_rates:u1_rate2',
                        'parameters:c', 'parameters:u1']:
            self.assertTrue(os.path.exists(f'plots/{varname.replace(":","_")}.png'))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
