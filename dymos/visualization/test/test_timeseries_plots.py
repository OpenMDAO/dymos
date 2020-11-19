import os
import unittest
import warnings

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.visualization.timeseries_plots import timeseries_plots


# @use_tempdirs
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

        # solution_record_file = 'test_timeseries_plots.db'
        # simulation_record_file = 'dymos_solution.db'

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

    def test_brachistochrone_timeseries_plots_solution_only_set_solution_record_file(self):
        dm.run_problem(self.p,make_plots=False, solution_record_file='solution_record_file.db' # will plot with timeseries_plots function
                       ) # records to the hardcode file 'dymos_simulation.db'

        timeseries_plots('solution_record_file.db')
        self.assertTrue(os.path.exists('plots/test_x.png'))
        self.assertTrue(os.path.exists('plots/test_y.png'))
        self.assertTrue(os.path.exists('plots/test_v.png'))

    def test_brachistochrone_timeseries_plots_solution_and_simulation(self):
        dm.run_problem(self.p, simulate=True, make_plots=False, # will plot with timeseries_plots function
                       simulation_record_file='simulation_record_file.db') # records to the hardcode file 'dymos_simulation.db'

        timeseries_plots('dymos_solution.db', plot_simulation=True,
                         simulation_record_file='simulation_record_file.db')



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

    def test_brachistochrone_timeseries_plots_set_plot_dir(self):
        dm.run_problem(self.p, make_plots=False)

        plot_dir = "test_plot_dir"
        timeseries_plots('dymos_solution.db',plot_dir=plot_dir)

        self.assertTrue(os.path.exists(os.path.join(plot_dir,'test_x.png')))
        self.assertTrue(os.path.exists(os.path.join(plot_dir,'test_y.png')))
        self.assertTrue(os.path.exists(os.path.join(plot_dir,'test_v.png')))


# @use_tempdirs
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

    # def test_timeseries_plots(self):
    #     burn1_accel = self.p.get_val('burn1.states:accel')
    #     burn2_accel = self.p.get_val('burn2.states:accel')
    #     accel_link_error = self.p.get_val('linkages.burn1|burn2_accel')
    #     assert_near_equal(accel_link_error, burn2_accel[0]-burn1_accel[-1])
    #
    #     from dymos.visualization.timeseries_plots import timeseries_plots
    #
    #     timeseries_plots('test_timeseries_plots.db')

    def test_timeseries_from_case_recorder_with_simulate(self):
        pass

        # In order to make obtaining the timeseries output of a phase easier, each phase
        # provides a timeseries component which collects and outputs the appropriate
        # timeseries data. For the pseudospectral transcriptions, timeseries outputs are
        # provided at all nodes. For the RungeKutta transcription, timeseries outputs
        # are provided at the segment endpoints. By default, the timeseries output will
        # include the following variables for every problem.
        #

# @use_tempdirs
class TestTwoPhaseCannonballForDocs(unittest.TestCase):

    def test_two_phase_cannonball_make_plot(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.cannonball.size_comp import CannonballSizeComp
        from dymos.examples.cannonball.cannonball_phase import CannonballPhase

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        external_params = p.model.add_subsystem('external_params', om.IndepVarComp())

        external_params.add_output('radius', val=0.10, units='m')
        external_params.add_output('dens', val=7.87, units='g/cm**3')

        external_params.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10)

        p.model.add_subsystem('size_comp', CannonballSizeComp())

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = CannonballPhase(transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.set_state_options('r', fix_initial=True, fix_final=False)
        ascent.set_state_options('h', fix_initial=True, fix_final=False)
        ascent.set_state_options('gam', fix_initial=False, fix_final=True)
        ascent.set_state_options('v', fix_initial=False, fix_final=False)

        ascent.add_parameter('S', targets=['aero.S'], units='m**2')
        ascent.add_parameter('mass', targets=['eom.m', 'kinetic_energy.m'], units='kg')

        # Limit the muzzle energy
        ascent.add_boundary_constraint('kinetic_energy.ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000, shape=(1,))

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = CannonballPhase(transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', )
        descent.add_state('h', fix_initial=False, fix_final=True)
        descent.add_state('gam', fix_initial=False, fix_final=False)
        descent.add_state('v', fix_initial=False, fix_final=False)

        descent.add_parameter('S', targets=['aero.S'], units='m**2')
        descent.add_parameter('mass', targets=['eom.m', 'kinetic_energy.m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['aero.CD'], 'descent': ['aero.CD']},
                           val=0.5, units=None, opt=False)
        traj.add_parameter('CL',
                           targets={'ascent': ['aero.CL'], 'descent': ['aero.CL']},
                           val=0.0, units=None, opt=False)
        traj.add_parameter('T',
                           targets={'ascent': ['eom.T'], 'descent': ['eom.T']},
                           val=0.0, units='N', opt=False)
        traj.add_parameter('alpha',
                           targets={'ascent': ['eom.alpha'], 'descent': ['eom.alpha']},
                           val=0.0, units='deg', opt=False)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'mass', 'descent': 'mass'})

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # A linear solver at the top level can improve performance.
        p.model.linear_solver = om.DirectSolver()

        # Finish Problem Setup
        p.setup()

        # Set Initial Guesses
        p.set_val('external_params.radius', 0.05, units='m')
        p.set_val('external_params.dens', 7.87, units='g/cm**3')

        p.set_val('traj.parameters:CD', 0.5)
        p.set_val('traj.parameters:CL', 0.0)
        p.set_val('traj.parameters:T', 0.0)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interpolate(ys=[0, 100], nodes='state_input'))
        p.set_val('traj.ascent.states:h', ascent.interpolate(ys=[0, 100], nodes='state_input'))
        p.set_val('traj.ascent.states:v', ascent.interpolate(ys=[200, 150], nodes='state_input'))
        p.set_val('traj.ascent.states:gam', ascent.interpolate(ys=[25, 0], nodes='state_input'),
                  units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interpolate(ys=[100, 200], nodes='state_input'))
        p.set_val('traj.descent.states:h', descent.interpolate(ys=[100, 0], nodes='state_input'))
        p.set_val('traj.descent.states:v', descent.interpolate(ys=[150, 200], nodes='state_input'))
        p.set_val('traj.descent.states:gam', descent.interpolate(ys=[0, -45], nodes='state_input'),
                  units='deg')

        # dm.run_problem(p)
        #
        # timeseries_plots('dymos_solution.db')

        dm.run_problem(p, simulate=True, make_plots=False, # will plot with timeseries_plots function
                       simulation_record_file='simulation_record_file.db') # records to the hardcode file 'dymos_simulation.db'

        timeseries_plots('dymos_solution.db', plot_simulation=True,
                         simulation_record_file='simulation_record_file.db')


        # exp_out = traj.simulate()
        #
        # import matplotlib.pyplot as plt
        # # plt.switch_backend('Agg')
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        #
        # time_imp = {'ascent': p.get_val('traj.ascent.timeseries.time'),
        #             'descent': p.get_val('traj.descent.timeseries.time')}
        #
        # time_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.time'),
        #             'descent': exp_out.get_val('traj.descent.timeseries.time')}
        #
        # r_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:r'),
        #          'descent': p.get_val('traj.descent.timeseries.states:r')}
        #
        # r_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:r'),
        #          'descent': exp_out.get_val('traj.descent.timeseries.states:r')}
        #
        # h_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:h'),
        #          'descent': p.get_val('traj.descent.timeseries.states:h')}
        #
        # h_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:h'),
        #          'descent': exp_out.get_val('traj.descent.timeseries.states:h')}
        #
        # axes.plot(r_imp['ascent'], h_imp['ascent'], 'bo')
        #
        # axes.plot(r_imp['descent'], h_imp['descent'], 'ro')
        #
        # axes.plot(r_exp['ascent'], h_exp['ascent'], 'b--')
        #
        # axes.plot(r_exp['descent'], h_exp['descent'], 'r--')
        #
        # axes.set_xlabel('range (m)')
        # axes.set_ylabel('altitude (m)')
        #
        # fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
        # states = ['r', 'h', 'v', 'gam']
        # for i, state in enumerate(states):
        #     x_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:{0}'.format(state)),
        #              'descent': p.get_val('traj.descent.timeseries.states:{0}'.format(state))}
        #
        #     x_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:{0}'.format(state)),
        #              'descent': exp_out.get_val('traj.descent.timeseries.states:{0}'.format(state))}
        #
        #     axes[i].set_ylabel(state)
        #
        #     axes[i].plot(time_imp['ascent'], x_imp['ascent'], 'bo')
        #     axes[i].plot(time_imp['descent'], x_imp['descent'], 'ro')
        #     axes[i].plot(time_exp['ascent'], x_exp['ascent'], 'b--')
        #     axes[i].plot(time_exp['descent'], x_exp['descent'], 'r--')
        #
        # params = ['CL', 'CD', 'T', 'alpha', 'mass', 'S']
        # fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6))
        # for i, param in enumerate(params):
        #     p_imp = {
        #         'ascent': p.get_val('traj.ascent.timeseries.parameters:{0}'.format(param)),
        #         'descent': p.get_val('traj.descent.timeseries.parameters:{0}'.format(param))}
        #
        #     p_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.'
        #                                        'parameters:{0}'.format(param)),
        #              'descent': exp_out.get_val('traj.descent.timeseries.'
        #                                         'parameters:{0}'.format(param))}
        #
        #     axes[i].set_ylabel(param)
        #
        #     axes[i].plot(time_imp['ascent'], p_imp['ascent'], 'bo')
        #     axes[i].plot(time_imp['descent'], p_imp['descent'], 'ro')
        #     axes[i].plot(time_exp['ascent'], p_exp['ascent'], 'b--')
        #     axes[i].plot(time_exp['descent'], p_exp['descent'], 'r--')
        #
        # plt.show()

    def test_trajectory_linked_phases_make_plot(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # p = self.p = om.Problem(model=om.Group())
        # self.traj = p.model.add_subsystem('traj', dm.Trajectory())


        # Since we're only testing features like get_values that don't rely on a converged
        # solution, no driver is attached.  We'll just invoke run_model.

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.declare_coloring()

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

        # p.run_model()
        dm.run_problem(p, simulate=True, make_plots=False, # will plot with timeseries_plots function
                       simulation_record_file='simulation_record_file.db') # records to the hardcode file 'dymos_simulation.db'

        timeseries_plots('dymos_solution.db', plot_simulation=True,
                         simulation_record_file='simulation_record_file.db')



if __name__ == '__main__':  # pragma: no cover
    unittest.main()
