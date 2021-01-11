import os
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from openmdao.utils.general_utils import set_pyoptsparse_opt
_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)

from openmdao.utils.testing_utils import use_tempdirs
from dymos.utils.doc_utils import save_for_docs


@use_tempdirs
class TestBrachistochroneForDocs(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @save_for_docs
    def test_brachistochrone_partials(self):
        import numpy as np
        import openmdao.api as om
        from dymos.utils.testing_utils import assert_check_partials
        from dymos.examples.brachistochrone.doc.brachistochrone_ode import BrachistochroneODE

        num_nodes = 5

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('vars', om.IndepVarComp())
        ivc.add_output('v', shape=(num_nodes,), units='m/s')
        ivc.add_output('theta', shape=(num_nodes,), units='deg')

        p.model.add_subsystem('ode', BrachistochroneODE(num_nodes=num_nodes))

        p.model.connect('vars.v', 'ode.v')
        p.model.connect('vars.theta', 'ode.theta')

        p.setup(force_alloc_complex=True)

        p.set_val('vars.v', 10*np.random.random(num_nodes))
        p.set_val('vars.theta', 10*np.random.uniform(1, 179, num_nodes))

        p.run_model()
        cpd = p.check_partials(method='cs', compact_print=True)
        assert_check_partials(cpd)

    @save_for_docs
    def test_brachistochrone_for_docs_gauss_lobatto(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE
        import matplotlib.pyplot as plt

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
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

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)
        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

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
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
                       'x (m)', 'y (m)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:theta',
                       'time (s)', 'theta (deg)')],
                     title='Brachistochrone Solution\nHigh-Order Gauss-Lobatto Method',
                     p_sol=p, p_sim=exp_out)

        plt.show()

    @save_for_docs
    def test_brachistochrone_for_docs_radau(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
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

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

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
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
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

    @save_for_docs
    def test_brachistochrone_for_docs_runge_kutta(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
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

        phase.add_state('x', fix_initial=True, fix_final=False)

        phase.add_state('y', fix_initial=True, fix_final=False)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

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

        p.model.linear_solver = om.DirectSolver()

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
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
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

    @save_for_docs
    def test_brachistochrone_for_docs_runge_kutta_polynomial_controls(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
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

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_polynomial_control('theta', order=1,
                                     units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

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

        p.model.linear_solver = om.DirectSolver()

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
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
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

    @unittest.skipIf(optimizer != 'IPOPT', 'IPOPT not available')
    def test_brachistochrone_for_docs_coloring_demo(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE
        from openmdao.utils.general_utils import set_pyoptsparse_opt

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        _, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)
        p.driver = om.pyOptSparseDriver(optimizer=optimizer)
        p.driver.declare_coloring(tol=1.0E-12)

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        #
        # In this case the phase has many segments to demonstrate the impact of coloring.
        #
        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.Radau(num_segments=100)))

        #
        # Set the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

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
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
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

    @unittest.skipIf(optimizer != 'IPOPT', 'IPOPT not available')
    def test_brachistochrone_for_docs_coloring_demo_solve_segments(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.brachistochrone import BrachistochroneODE
        from openmdao.utils.general_utils import set_pyoptsparse_opt

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        _, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)
        p.driver = om.pyOptSparseDriver(optimizer=optimizer)
        p.driver.opt_settings['print_level'] = 4
        # p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        #
        # In this case the phase has many segments to demonstrate the impact of coloring.
        #
        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.Radau(num_segments=100,
                                                               solve_segments='forward')))

        #
        # Set the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True)

        phase.add_state('y', fix_initial=True)

        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

        #
        # Replace state terminal bounds with nonlinear constraints
        #
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

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
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
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

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
