from __future__ import print_function, absolute_import, division

import os
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class TestBrachistochroneForDocs(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_for_docs_gauss_lobatto(self):
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, \
            SqliteRecorder, CaseReader
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.GaussLobatto(num_segments=10)))

        #
        # Set the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))
        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)
        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)
        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj.phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        #
        # Solve for the optimal trajectory
        #
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
                       'x (m)', 'y (m)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:theta',
                       'time (s)', 'theta (deg)')],
                     title='Brachistochrone Solution\nHigh-Order Gauss-Lobatto Method',
                     p_sol=p, p_sim=exp_out)

        plt.show()

    def test_brachistochrone_for_docs_radau(self):
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, SqliteRecorder
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.Radau(num_segments=10)))

        #
        # Set the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))
        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)
        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)
        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj.phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        #
        # Solve for the optimal trajectory
        #
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
                         tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
                       'x (m)', 'y (m)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:theta',
                       'time (s)', 'theta (deg)')],
                     title='Brachistochrone Solution\nRadau Pseudospectral Method',
                     p_sol=p, p_sim=exp_out)

        plt.show()

    def test_brachistochrone_for_docs_runge_kutta(self):
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, SqliteRecorder
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.RungeKutta(num_segments=10)))

        #
        # Set the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))
        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)
        phase.add_control('theta', units='deg', lower=0.01, upper=179.9)
        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        #
        # Final state values are not optimization variables, so we must enforce final values
        # with boundary constraints, not simple bounds.
        #
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj.phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        #
        # Solve for the optimal trajectory
        #
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
                         tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
                       'x (m)', 'y (m)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:theta',
                       'time (s)', 'theta (deg)')],
                     title='Brachistochrone Solution\nRK4 Shooting Method',
                     p_sol=p, p_sim=exp_out)

        plt.show()

    def test_brachistochrone_for_docs_runge_kutta_polynomial_controls(self):
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, SqliteRecorder
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.RungeKutta(num_segments=10)))

        #
        # Set the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))
        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)
        phase.add_polynomial_control('theta', units='deg', lower=0.01, upper=179.9, order=1)
        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        #
        # Final state values are not optimization variables, so we must enforce final values
        # with boundary constraints, not simple bounds.
        #
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj.phase0.polynomial_controls:theta'][:] = 5.0

        #
        # Solve for the optimal trajectory
        #
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
                         tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
                       'x (m)', 'y (m)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.polynomial_controls:theta',
                       'time (s)', 'theta (deg)')],
                     title='Brachistochrone Solution\nRK4 Shooting and Polynomial Controls',
                     p_sol=p, p_sim=exp_out)

        plt.show()


if __name__ == '__main__':
    unittest.main()
