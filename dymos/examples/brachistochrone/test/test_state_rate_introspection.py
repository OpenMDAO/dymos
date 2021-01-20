import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from dymos.utils.testing_utils import assert_timeseries_near_equal

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestIntegrateControl(unittest.TestCase):

    def _test_integrate_control(self, transcription):

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
                         transcription=transcription(num_segments=10, order=3))

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

    def _test_integrate_control_rate(self, transcription):

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
                         transcription=transcription(num_segments=10, order=3))

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

        # Force the initial value of the theta polynomial control to equal the initial value of the theta state.
        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='theta', var_b='int_theta',
                                    loc_a='initial', loc_b='initial')

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

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sim, int_theta_sim, t_sim, theta_sim, tolerance=1.0E-2)

    def _test_integrate_control_rate2(self, transcription):
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
                         transcription=transcription(num_segments=10, order=3))

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

        # Force the initial value of the theta polynomial control to equal the initial value of the theta state.
        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='theta', var_b='int_theta',
                                    loc_a='initial', loc_b='initial')

        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='int_theta_dot', var_b='theta_rate',
                                    loc_a='initial', loc_b='initial',
                                    units='rad/s')

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

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, tolerance=1.0E-2)

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sim, int_theta_sim, t_sim, theta_sim, tolerance=1.0E-2)

    def test_integrate_control_gl(self):
        self._test_integrate_control(dm.GaussLobatto)

    def test_integrate_control_radau(self):
        self._test_integrate_control(dm.Radau)

    def test_integrate_control_rate_gl(self):
        self._test_integrate_control_rate(dm.GaussLobatto)

    def test_integrate_control_rate_radau(self):
        self._test_integrate_control_rate(dm.Radau)

    def test_integrate_control_rate2_gl(self):
        self._test_integrate_control_rate2(dm.GaussLobatto)

    def test_integrate_control_rate2_radau(self):
        self._test_integrate_control_rate2(dm.Radau)


@use_tempdirs
class TestIntegratePolynomialControl(unittest.TestCase):

    def _test_integrate_polynomial_control(self, transcription):
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
                         transcription=transcription(num_segments=20, order=3))

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

        phase.add_state('int_theta', fix_initial=False, rate_source='theta_rate',
                        targets=['theta'])

        # Define theta as a control.
        phase.add_polynomial_control(name='theta_rate', order=11, units='rad/s', shape=(1,), targets=None)

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

        p.set_val('traj.phase0.polynomial_controls:theta_rate', 10.0, units='deg/s')

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

        theta_rate_sol = sol.get_val('traj.phase0.timeseries.polynomial_controls:theta_rate')
        theta_rate_sim = sim.get_val('traj.phase0.timeseries.polynomial_controls:theta_rate')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, theta_rate_sol, t_sim, theta_rate_sim, tolerance=1.0E-2)

    def _test_integrate_polynomial_control_rate(self, transcription):
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
                         transcription=transcription(num_segments=10, order=5))

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

        phase.add_state('int_theta', fix_initial=False, rate_source='theta_rate',
                        targets=['theta'])

        # Define theta as a control.
        phase.add_polynomial_control(name='theta', order=11, units='rad', shape=(1,), targets=None)

        # Force the initial value of the theta polynomial control to equal the initial value of the theta state.
        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='theta', var_b='int_theta',
                                    loc_a='initial', loc_b='initial')

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

        p.set_val('traj.phase0.polynomial_controls:theta', 45, units='deg')

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

        theta_sol = sol.get_val('traj.phase0.timeseries.polynomial_controls:theta')
        theta_sim = sim.get_val('traj.phase0.timeseries.polynomial_controls:theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim,
                                     tolerance=1.0E-3)

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, tolerance=1.0E-3)
        # assert_timeseries_near_equal(t_sol, int_theta_sim, t_sol, theta_sim, tolerance=1.0E-3)

    def _test_integrate_polynomial_control_rate2(self, transcription):
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
                         transcription=transcription(num_segments=20, order=3))

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
        phase.add_state('int_theta', fix_initial=False, rate_source='int_theta_dot',
                        targets=['theta'])

        # Define theta as a control.
        phase.add_polynomial_control(name='theta', order=11, units='rad', shape=(1,), targets=None)

        # Force the initial value of the theta polynomial control to equal the initial value of the theta state.
        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='theta', var_b='int_theta',
                                    loc_a='initial', loc_b='initial')

        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='int_theta_dot', var_b='theta_rate',
                                    loc_a='initial', loc_b='initial',
                                    units='rad/s')

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

        p.set_val('traj.phase0.polynomial_controls:theta', 45, units='deg')

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

        theta_sol = sol.get_val('traj.phase0.timeseries.polynomial_controls:theta')
        theta_sim = sim.get_val('traj.phase0.timeseries.polynomial_controls:theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, tolerance=1.0E-2)

    def test_integrate_polynomial_control_gl(self):
        self._test_integrate_polynomial_control(dm.GaussLobatto)

    def test_integrate_polynomial_control_radau(self):
        self._test_integrate_polynomial_control(dm.Radau)

    def test_integrate_polynomial_control_rate_gl(self):
        self._test_integrate_polynomial_control_rate(dm.GaussLobatto)

    def test_integrate_polynomial_control_rate_radau(self):
        self._test_integrate_polynomial_control_rate(dm.Radau)

    def test_integrate_polynomial_control_rate2_gl(self):
        self._test_integrate_polynomial_control_rate2(dm.GaussLobatto)

    def test_integrate_polynomial_control_rate2_radau(self):
        self._test_integrate_polynomial_control_rate2(dm.Radau)


@use_tempdirs
class TestIntegrateTimeParamAndState(unittest.TestCase):

    def _test_transcription(self, transcription=dm.GaussLobatto):
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
                         transcription=transcription(num_segments=10, order=3,
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

        time_sol = p.get_val('traj.phase0.timeseries.time')
        time_phase_sol = p.get_val('traj.phase0.timeseries.time_phase')
        int_one_sol = p.get_val('traj.phase0.timeseries.states:int_one')
        int_time_sol = p.get_val('traj.phase0.timeseries.states:int_time')
        int_time_phase_sol = p.get_val('traj.phase0.timeseries.states:int_time_phase')
        int_int_one_sol = p.get_val('traj.phase0.timeseries.states:int_int_one')

        time_sim = p.get_val('traj.phase0.timeseries.time')
        time_phase_sim = p.get_val('traj.phase0.timeseries.time_phase')
        int_one_sim = p.get_val('traj.phase0.timeseries.states:int_one')
        int_time_sim = p.get_val('traj.phase0.timeseries.states:int_time')
        int_time_phase_sim = p.get_val('traj.phase0.timeseries.states:int_time_phase')
        int_int_one_sim = p.get_val('traj.phase0.timeseries.states:int_int_one')

        # Integral of one should match time and time_phase in this case.
        assert_near_equal(int_one_sol, time_sol, tolerance=1.0E-12)
        assert_near_equal(int_one_sol, time_phase_sol, tolerance=1.0E-12)

        assert_near_equal(int_one_sim, time_sim, tolerance=1.0E-12)
        assert_near_equal(int_one_sim, time_phase_sim, tolerance=1.0E-12)

        # Integral of time and time_phase should be t**2/2
        assert_near_equal(time_sol, time_phase_sol, tolerance=1.0E-12)
        assert_near_equal(int_time_sol, time_sol**2/2, tolerance=1.0E-12)
        assert_near_equal(int_time_phase_sol, time_phase_sol**2/2, tolerance=1.0E-12)

        assert_near_equal(time_sim, time_phase_sim, tolerance=1.0E-12)
        assert_near_equal(int_time_sim, time_sim**2/2, tolerance=1.0E-12)
        assert_near_equal(int_time_phase_sim, time_phase_sim**2/2, tolerance=1.0E-12)

        # Double integral of one should be the same as the integral of time
        assert_near_equal(int_int_one_sol, int_time_sol, tolerance=1.0E-12)
        assert_near_equal(int_int_one_sim, int_time_sim, tolerance=1.0E-12)

        assert_timeseries_near_equal(time_sol, int_int_one_sol, time_sim, int_int_one_sim)

    def test_gl(self):
        self._test_transcription(transcription=dm.GaussLobatto)

    def test_radau(self):
        self._test_transcription(transcription=dm.Radau)
