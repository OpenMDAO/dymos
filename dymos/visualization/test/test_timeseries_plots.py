import os
import unittest
import warnings

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.visualization.timeseries_plots import timeseries_plots


class TestSimpleTimeSeriesPlots(unittest.TestCase):
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

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        units=BrachistochroneODE.states['x']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)
        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        units=BrachistochroneODE.states['y']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)
        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        units=BrachistochroneODE.states['v']['units'],
                        fix_initial=True, fix_final=False, solve_segments=False)

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

        p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['traj0.phase0.parameters:g'] = 9.80665

        solution_record_file = 'test_timeseries_plots.db'
        simulation_record_file = 'dymos_solution.db'

        # self.recorder = om.SqliteRecorder(solution_record_file)

        # p.model.add_recorder(self.recorder)

        p.setup()

        self.p = p

    def tearDown(self):
        for filename in ['test_time_series_plots.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_timeseries_plots(self):
        dm.run_problem(self.p, make_plots=False)

        timeseries_plots('dymos_solution.db')
        self.assertTrue(os.path.exists('plots/test_x.png'))
        self.assertTrue(os.path.exists('plots/test_y.png'))
        self.assertTrue(os.path.exists('plots/test_v.png'))

    def test_brachistochrone_timeseries_plots_solution_and_simulation(self):
        dm.run_problem(self.p,simulate=True, make_plots=False, # will plot with timeseries_plots function
                       simulation_record_file='simulation_record_file.db') # records to the hardcode file 'dymos_simulation.db'

        timeseries_plots('dymos_solution.db', plot_simulation=True, simulation_record_file='simulation_record_file.db')

    def test_brachistochrone_timeseries_plots_solution_only(self):
        dm.run_problem(self.p,make_plots=False, # will plot with timeseries_plots function
                       ) # records to the hardcode file 'dymos_simulation.db'

        timeseries_plots('dymos_solution.db', plot_simulation=False)

    def test_brachistochrone_timeseries_plots_plot_simulation_true_but_no_path_given(self):
        dm.run_problem(self.p, make_plots=False)
        with self.assertRaises(ValueError) as e:
            timeseries_plots('dymos_solution.db', plot_simulation=True)
        expected = 'If plot_simulation is True, simulation_record_file must be path to simulation case recorder file, not None'
        self.assertEqual(str(e.exception), expected)

    def test_brachistochrone_timeseries_plots_plot_simulation_false_but_path_given(self):
        dm.run_problem(self.p, make_plots=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            timeseries_plots('dymos_solution.db', simulation_record_file='simulation_record_file.db')

        expected = 'Setting simulation_record_file but not setting plot_simulation will not result in plotting simulation data'

        self.assertIn(expected, [str(ww.message) for ww in w])

class TestTimeSeriesPlots(unittest.TestCase):

    def tearDown(self):
        for filename in ['test_time_series_plots.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def setUp(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # Since we're only testing features like get_values that don't rely on a converged
        # solution, no driver is attached.  We'll just invoke run_model.

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

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

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
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

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
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

        self.recorder = om.SqliteRecorder('test_timeseries_plots.db')

        p.model.add_recorder(self.recorder)

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interpolate(ys=[1, 1.5], nodes='state_input'))
        p.set_val('burn1.states:theta', value=burn1.interpolate(ys=[0, 1.7], nodes='state_input'))
        p.set_val('burn1.states:vr', value=burn1.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('burn1.states:vt', value=burn1.interpolate(ys=[1, 1], nodes='state_input'))
        p.set_val('burn1.states:accel', value=burn1.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1], nodes='state_input'))
        p.set_val('burn1.controls:u1',
                  value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interpolate(ys=[1.3, 1.5], nodes='state_input'))
        p.set_val('coast.states:theta',
                  value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))
        p.set_val('coast.states:vr', value=coast.interpolate(ys=[0.3285, 0], nodes='state_input'))
        p.set_val('coast.states:vt', value=coast.interpolate(ys=[0.97, 1], nodes='state_input'))
        p.set_val('coast.states:accel', value=coast.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('coast.controls:u1', value=coast.interpolate(ys=[0, 0], nodes='control_input'))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interpolate(ys=[1, 3], nodes='state_input'))
        p.set_val('burn2.states:theta', value=burn2.interpolate(ys=[0, 4.0], nodes='state_input'))
        p.set_val('burn2.states:vr', value=burn2.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('burn2.states:vt',
                  value=burn2.interpolate(ys=[1, np.sqrt(1 / 3)], nodes='state_input'))
        p.set_val('burn2.states:accel', value=burn2.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('burn2.states:deltav',
                  value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
        p.set_val('burn2.controls:u1', value=burn2.interpolate(ys=[1, 1], nodes='control_input'))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

    def test_timeseries_plots(self):
        burn1_accel = self.p.get_val('burn1.states:accel')
        burn2_accel = self.p.get_val('burn2.states:accel')
        accel_link_error = self.p.get_val('linkages.burn1|burn2_accel')
        assert_near_equal(accel_link_error, burn2_accel[0]-burn1_accel[-1])

        from dymos.visualization.timeseries_plots import timeseries_plots

        timeseries_plots('test_timeseries_plots.db')

    def test_timeseries_from_case_recorder_with_simulate(self):
        pass

        # In order to make obtaining the timeseries output of a phase easier, each phase
        # provides a timeseries component which collects and outputs the appropriate
        # timeseries data. For the pseudospectral transcriptions, timeseries outputs are
        # provided at all nodes. For the RungeKutta transcription, timeseries outputs
        # are provided at the segment endpoints. By default, the timeseries output will
        # include the following variables for every problem.
        #



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
