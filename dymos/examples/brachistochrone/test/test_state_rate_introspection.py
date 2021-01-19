import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.testing_utils import assert_timeseries_near_equal

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestBrachistochroneQuickStart(unittest.TestCase):

    def test_integrate_control(self):

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(1.0, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_state('int_theta', fix_initial=False, rate_source='theta_rate', targets=['theta'])

        # Define theta as a control.
        phase.add_control(name='theta_rate', units='rad/s', shape=(1,), targets=None)

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        p.set_val('traj.phase0.t_initial', 0.0, units='s')
        p.set_val('traj.phase0.t_duration', 5.0, units='s')

        p.set_val('traj.phase0.states:x',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:y',
                  phase.interpolate(ys=[10, 5], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:v',
                  phase.interpolate(ys=[0, 5], nodes='state_input'),
                  units='m/s')

        p.set_val('traj.phase0.states:int_theta',
                  phase.interpolate(ys=[0.1, 45], nodes='state_input'),
                  units='deg')

        # Additional states to test rate sources
        p.set_val('traj.phase0.controls:theta_rate',
                  phase.interpolate(ys=[10, 10], nodes='control_input'),
                  units='deg/s')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=False)

        sol = om.CaseReader('dymos_solution.db').get_case('final')
        sim = om.CaseReader('dymos_simulation.db').get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.states:x')
        x_sim = sim.get_val('traj.phase0.timeseries.states:x')

        y_sol = sol.get_val('traj.phase0.timeseries.states:y')
        y_sim = sim.get_val('traj.phase0.timeseries.states:y')

        v_sol = sol.get_val('traj.phase0.timeseries.states:v')
        v_sim = sim.get_val('traj.phase0.timeseries.states:v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.states:int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.states:int_theta')

        theta_rate_sol = sol.get_val('traj.phase0.timeseries.controls:theta_rate')
        theta_rate_sim = sim.get_val('traj.phase0.timeseries.controls:theta_rate')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, theta_rate_sol, t_sim, theta_rate_sim, tolerance=1.0E-3)


    def test_integrate_control_rate(self):

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=5))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(1.0, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_state('int_theta', fix_initial=False, rate_source='theta_rate', targets=['theta'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', shape=(1,), targets=None)

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        p.set_val('traj.phase0.t_initial', 0.0, units='s')
        p.set_val('traj.phase0.t_duration', 5.0, units='s')

        p.set_val('traj.phase0.states:x',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:y',
                  phase.interpolate(ys=[10, 5], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:v',
                  phase.interpolate(ys=[0, 5], nodes='state_input'),
                  units='m/s')

        p.set_val('traj.phase0.states:int_theta',
                  phase.interpolate(ys=[0.1, 45], nodes='state_input'),
                  units='deg')

        # Additional states to test rate sources
        p.set_val('traj.phase0.controls:theta',
                  phase.interpolate(ys=[0, 100], nodes='control_input'),
                  units='deg')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=False)

        sol = om.CaseReader('dymos_solution.db').get_case('final')
        sim = om.CaseReader('dymos_simulation.db').get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.states:x')
        x_sim = sim.get_val('traj.phase0.timeseries.states:x')

        y_sol = sol.get_val('traj.phase0.timeseries.states:y')
        y_sim = sim.get_val('traj.phase0.timeseries.states:y')

        v_sol = sol.get_val('traj.phase0.timeseries.states:v')
        v_sim = sim.get_val('traj.phase0.timeseries.states:v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.states:int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.states:int_theta')

        theta_sol = sol.get_val('traj.phase0.timeseries.controls:theta')
        theta_sim = sim.get_val('traj.phase0.timeseries.controls:theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, tolerance=1.0E-3)

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, int_theta_sim, t_sol, theta_sim, tolerance=1.0E-3)

    def test_integrate_control_rate2(self):

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(1.0, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_state('int_theta_dot', fix_initial=False, rate_source='theta_rate2')
        phase.add_state('int_theta', fix_initial=False, rate_source='int_theta_dot', targets=['theta'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', shape=(1,), targets=None)

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        p.set_val('traj.phase0.t_initial', 0.0, units='s')
        p.set_val('traj.phase0.t_duration', 5.0, units='s')

        p.set_val('traj.phase0.states:x',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:y',
                  phase.interpolate(ys=[10, 5], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:v',
                  phase.interpolate(ys=[0, 5], nodes='state_input'),
                  units='m/s')

        p.set_val('traj.phase0.states:int_theta',
                  phase.interpolate(ys=[0.1, 45], nodes='state_input'),
                  units='deg')

        p.set_val('traj.phase0.states:int_theta_dot',
                  phase.interpolate(ys=[0.0, 0.0], nodes='state_input'),
                  units='deg/s')

        # Additional states to test rate sources
        p.set_val('traj.phase0.controls:theta',
                  phase.interpolate(ys=[0, 100], nodes='control_input'),
                  units='deg')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=True)

        sol = om.CaseReader('dymos_solution.db').get_case('final')
        sim = om.CaseReader('dymos_simulation.db').get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.states:x')
        x_sim = sim.get_val('traj.phase0.timeseries.states:x')

        y_sol = sol.get_val('traj.phase0.timeseries.states:y')
        y_sim = sim.get_val('traj.phase0.timeseries.states:y')

        v_sol = sol.get_val('traj.phase0.timeseries.states:v')
        v_sim = sim.get_val('traj.phase0.timeseries.states:v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.states:int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.states:int_theta')

        theta_sol = sol.get_val('traj.phase0.timeseries.controls:theta')
        theta_sim = sim.get_val('traj.phase0.timeseries.controls:theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, tolerance=1.0E-2)

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sim, t_sol, theta_sim, tolerance=1.0E-2)

    def test_integrate_time_parameter_and_state(self):
        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3,
                                                       solve_segments='forward'))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True)
        phase.add_state('y', fix_initial=True)
        phase.add_state('v', fix_initial=True)

        phase.add_state('int_one', fix_initial=True, rate_source='one')
        phase.add_state('int_time', fix_initial=True, rate_source='time')
        phase.add_state('int_time_phase', fix_initial=True, rate_source='time_phase')
        phase.add_state('int_int_one', fix_initial=True, rate_source='int_one')

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi)

        # With no targets we must explicitly assign units and shape to this parameter.
        # Its only purpose is to be integrated as the rate source for a state.
        phase.add_parameter(name='one', opt=False, units=None, shape=(1,))

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        p.set_val('traj.phase0.t_initial', 0.0, units='s')
        p.set_val('traj.phase0.t_duration', 5.0, units='s')

        p.set_val('traj.phase0.parameters:one', 1.0)

        p.set_val('traj.phase0.states:x',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:y',
                  phase.interpolate(ys=[10, 5], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:v',
                  phase.interpolate(ys=[0, 5], nodes='state_input'),
                  units='m/s')

        p.set_val('traj.phase0.controls:theta',
                  phase.interpolate(ys=[0.1, 45], nodes='control_input'),
                  units='deg')

        # Additional states to test rate sources
        p.set_val('traj.phase0.states:int_one',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='s')

        p.set_val('traj.phase0.states:int_time',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='s**2')

        p.set_val('traj.phase0.states:int_time_phase',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='s**2')

        p.set_val('traj.phase0.states:int_int_one',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='s**2')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True)

        time = p.get_val('traj.phase0.timeseries.time')
        time_phase = p.get_val('traj.phase0.timeseries.time_phase')
        int_one = p.get_val('traj.phase0.timeseries.states:int_one')
        int_time = p.get_val('traj.phase0.timeseries.states:int_time')
        int_time_phase = p.get_val('traj.phase0.timeseries.states:int_time_phase')
        int_int_one = p.get_val('traj.phase0.timeseries.states:int_int_one')

        # Integral of one should match time and time_phase in this case.
        assert_near_equal(int_one, time)
        assert_near_equal(int_one, time_phase)

        # Integral of time and time_phase should be t**2/2
        assert_near_equal(time, time_phase)
        assert_near_equal(int_time, time**2/2)
        assert_near_equal(int_time_phase, time_phase**2/2)

        # Double integral of one should be the same as the integral of time
        assert_near_equal(int_int_one, int_time)
