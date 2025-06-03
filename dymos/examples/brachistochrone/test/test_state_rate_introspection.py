import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
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

        phase.set_time_val(initial=0.0, duration=5.0, units='s')

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')
        phase.set_state_val('int_theta', [0.1, 45], units='deg')
        phase.set_control_val('theta_rate', [10, 10], units='deg/s')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=False)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol = om.CaseReader(sol_db).get_case('final')
        sim = om.CaseReader(sim_db).get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.x')
        x_sim = sim.get_val('traj.phase0.timeseries.x')

        y_sol = sol.get_val('traj.phase0.timeseries.y')
        y_sim = sim.get_val('traj.phase0.timeseries.y')

        v_sol = sol.get_val('traj.phase0.timeseries.v')
        v_sim = sim.get_val('traj.phase0.timeseries.v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.int_theta')

        theta_rate_sol = sol.get_val('traj.phase0.timeseries.theta_rate')
        theta_rate_sim = sim.get_val('traj.phase0.timeseries.theta_rate')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sim, theta_rate_sim, t_sol, theta_rate_sol,  rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)

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

        phase.set_time_val(initial=0.0, duration=5.0, units='s')

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')
        phase.set_state_val('int_theta', [0.1, 45], units='deg')
        phase.set_control_val('theta', [0, 100], units='deg')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=False)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol = om.CaseReader(sol_db).get_case('final')
        sim = om.CaseReader(sim_db).get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.x')
        x_sim = sim.get_val('traj.phase0.timeseries.x')

        y_sol = sol.get_val('traj.phase0.timeseries.y')
        y_sim = sim.get_val('traj.phase0.timeseries.y')

        v_sol = sol.get_val('traj.phase0.timeseries.v')
        v_sim = sim.get_val('traj.phase0.timeseries.v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.int_theta')

        theta_sol = sol.get_val('traj.phase0.timeseries.theta')
        theta_sim = sim.get_val('traj.phase0.timeseries.theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sim, int_theta_sim, t_sim, theta_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)

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
                         transcription=transcription(num_segments=5, order=3))

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

        # Note we have to add theta_rate to the timeseries outputs here because
        # linkages get boundary values from that output.
        phase.add_timeseries_output('theta_rate')
        phase.add_timeseries_output('theta_rate2')

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
        phase.set_time_val(initial=0.0, duration=5.0, units='s')

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')
        phase.set_state_val('int_theta', [0.1, 45], units='deg')
        phase.set_state_val('int_theta_dot', [0.0, 0.0], units='deg/s')
        phase.set_control_val('theta', [0, 100], units='deg')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=False, simulate_kwargs={'atol': 1.0E-9, 'rtol': 1.0E-9,
                                                                            'times_per_seg': 10})

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol = om.CaseReader(sol_db).get_case('final')
        sim = om.CaseReader(sim_db).get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.x')
        x_sim = sim.get_val('traj.phase0.timeseries.x')

        y_sol = sol.get_val('traj.phase0.timeseries.y')
        y_sim = sim.get_val('traj.phase0.timeseries.y')

        v_sol = sol.get_val('traj.phase0.timeseries.v')
        v_sim = sim.get_val('traj.phase0.timeseries.v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.int_theta')

        theta_sol = sol.get_val('traj.phase0.timeseries.theta')
        theta_sim = sim.get_val('traj.phase0.timeseries.theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, rel_tolerance=1.0E-2, abs_tolerance=1.0E-1)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, rel_tolerance=1.0E-2, abs_tolerance=1.0E-1)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, rel_tolerance=1.0E-2, abs_tolerance=1.0E-1)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, rel_tolerance=1.0E-2,
                                     abs_tolerance=1.0E-1)

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, rel_tolerance=1.0E-2, abs_tolerance=1.0E-1)
        assert_timeseries_near_equal(t_sim, int_theta_sim, t_sim, theta_sim, rel_tolerance=1.0E-2, abs_tolerance=1.0E-1)

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
        phase.add_control(name='theta_rate', order=11, units='rad/s', shape=(1,), targets=None, control_type='polynomial')

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

        phase.set_time_val(initial=0.0, duration=5.0, units='s')

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')
        phase.set_state_val('int_theta', [0.1, 45], units='deg')
        phase.set_control_val('theta_rate', 10.0, units='deg/s')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol = om.CaseReader(sol_db).get_case('final')
        sim = om.CaseReader(sim_db).get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.x')
        x_sim = sim.get_val('traj.phase0.timeseries.x')

        y_sol = sol.get_val('traj.phase0.timeseries.y')
        y_sim = sim.get_val('traj.phase0.timeseries.y')

        v_sol = sol.get_val('traj.phase0.timeseries.v')
        v_sim = sim.get_val('traj.phase0.timeseries.v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.int_theta')

        theta_rate_sol = sol.get_val('traj.phase0.timeseries.theta_rate')
        theta_rate_sim = sim.get_val('traj.phase0.timeseries.theta_rate')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, rel_tolerance=4.0E-3,
                                     abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sim, theta_rate_sim, t_sol, theta_rate_sol, rel_tolerance=4.0E-3,
                                     abs_tolerance=1.0E-2)

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
        phase.add_control(name='theta', order=11, units='rad', shape=(1,), targets=None, control_type='polynomial')

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

        phase.set_time_val(initial=0.0, duration=5.0, units='s')

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')
        phase.set_state_val('int_theta', [0.1, 45], units='deg')
        phase.set_control_val('theta', 45.0, units='deg')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=False,
                       simulate_kwargs={'times_per_seg': 10})

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol = om.CaseReader(sol_db).get_case('final')
        sim = om.CaseReader(sim_db).get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.x')
        x_sim = sim.get_val('traj.phase0.timeseries.x')

        y_sol = sol.get_val('traj.phase0.timeseries.y')
        y_sim = sim.get_val('traj.phase0.timeseries.y')

        v_sol = sol.get_val('traj.phase0.timeseries.v')
        v_sim = sim.get_val('traj.phase0.timeseries.v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.int_theta')

        theta_sol = sol.get_val('traj.phase0.timeseries.theta')
        # theta_sim = sim.get_val('traj.phase0.timeseries.theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim,
                                     rel_tolerance=1.0E-2, abs_tolerance=1.5E-2)

        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sol, theta_sol, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)

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
        phase.add_control(name='theta', order=11, units='rad', shape=(1,), targets=None, control_type='polynomial')

        # Force the initial value of the theta polynomial control to equal the initial value of the theta state.
        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='theta', var_b='int_theta',
                                    loc_a='initial', loc_b='initial')

        traj.add_linkage_constraint(phase_a='phase0', phase_b='phase0',
                                    var_a='int_theta_dot', var_b='theta_rate',
                                    loc_a='initial', loc_b='initial',
                                    units='rad/s')

        # Note for this test we have to add theta rate to the timeseries outputs since it's
        # not there by default.
        phase.add_timeseries_output('theta_rate')

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

        phase.set_time_val(initial=0.0, duration=5.0, units='s')

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')
        phase.set_state_val('int_theta', [0.1, 45], units='deg')
        phase.set_state_val('int_theta_dot', [0.0, 0.0], units='deg/s')
        phase.set_control_val('theta', 45.0, units='deg')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True, make_plots=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol = om.CaseReader(sol_db).get_case('final')
        sim = om.CaseReader(sim_db).get_case('final')

        t_sol = sol.get_val('traj.phase0.timeseries.time')
        t_sim = sim.get_val('traj.phase0.timeseries.time')

        x_sol = sol.get_val('traj.phase0.timeseries.x')
        x_sim = sim.get_val('traj.phase0.timeseries.x')

        y_sol = sol.get_val('traj.phase0.timeseries.y')
        y_sim = sim.get_val('traj.phase0.timeseries.y')

        v_sol = sol.get_val('traj.phase0.timeseries.v')
        v_sim = sim.get_val('traj.phase0.timeseries.v')

        int_theta_sol = sol.get_val('traj.phase0.timeseries.int_theta')
        int_theta_sim = sim.get_val('traj.phase0.timeseries.int_theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)

        assert_timeseries_near_equal(t_sim, y_sim, t_sol, y_sol, rel_tolerance=8.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sim, v_sim, t_sol, v_sol, rel_tolerance=8.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sim, int_theta_sim, t_sol, int_theta_sol, rel_tolerance=8.0E-3, abs_tolerance=1.0E-2)
        assert_timeseries_near_equal(t_sol, int_theta_sol, t_sim, int_theta_sim, rel_tolerance=4.0E-3, abs_tolerance=1.0E-2)

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

    def _test_transcription(self, transcription=dm.GaussLobatto, time_name='time'):
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
        phase.set_time_options(fix_initial=True, fix_duration=True, units='s', name=time_name)

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
        phase.add_state('int_time', fix_initial=True, rate_source=time_name)
        phase.add_state('int_time_phase', fix_initial=True, rate_source=f'{time_name}_phase')
        phase.add_state('int_int_one', fix_initial=True, rate_source='int_one')

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi)

        # With no targets we must explicitly assign units and shape to this parameter.
        # Its only purpose is to be integrated as the rate source for a state.
        phase.add_parameter(name='one', opt=False, units=None, shape=(1,))

        # Minimize final time.
        phase.add_objective(time_name, loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        phase.set_time_val(initial=0.0, duration=5.0, units='s')

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')
        phase.set_state_val('int_one', [0, 10], units='s')
        phase.set_state_val('int_int_one', [0, 10], units='s**2')
        phase.set_state_val('int_time', [0, 10], units='s**2')
        phase.set_state_val('int_time_phase', [0, 10], units='s**2')
        phase.set_control_val('theta', [0.1, 45], units='deg')
        phase.set_parameter_val('one', 1.0)

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True)

        time_sol = p.get_val(f'traj.phase0.timeseries.{time_name}')
        time_phase_sol = p.get_val(f'traj.phase0.timeseries.{time_name}_phase')
        int_one_sol = p.get_val('traj.phase0.timeseries.int_one')
        int_time_sol = p.get_val('traj.phase0.timeseries.int_time')
        int_time_phase_sol = p.get_val('traj.phase0.timeseries.int_time_phase')
        int_int_one_sol = p.get_val('traj.phase0.timeseries.int_int_one')

        time_sim = p.get_val(f'traj.phase0.timeseries.{time_name}')
        time_phase_sim = p.get_val(f'traj.phase0.timeseries.{time_name}_phase')
        int_one_sim = p.get_val('traj.phase0.timeseries.int_one')
        int_time_sim = p.get_val('traj.phase0.timeseries.int_time')
        int_time_phase_sim = p.get_val('traj.phase0.timeseries.int_time_phase')
        int_int_one_sim = p.get_val('traj.phase0.timeseries.int_int_one')

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

        assert_timeseries_near_equal(time_sol, int_int_one_sol, time_sim, int_int_one_sim, rel_tolerance=1.0E-12)

    def test_integrated_times_params_and_states(self):
        for tx in (dm.GaussLobatto, dm.Radau):
            tx_name = 'GaussLobatto' if tx is dm.GaussLobatto else 'Radau'
            for time_name in ('time', 'elapsed_time'):
                with self.subTest(msg=f'{tx_name}: time_name=\'{time_name}\''):
                    self._test_transcription(transcription=tx, time_name=time_name)


@use_tempdirs
class TestInvalidStateRateSource(unittest.TestCase):

    def test_brach_invalid_state_rate_source(self):

        class _BrachODE(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                # Inputs
                self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
                self.add_input('g', val=9.80665 * np.ones(nn), desc='grav. acceleration', units='m/s/s')
                self.add_input('theta', val=np.ones(nn), desc='angle of wire', units='rad')

                self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')

                self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')

                self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

                self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                                units='m/s')

                self.declare_coloring(wrt='*', method='cs')

            def compute(self, inputs, outputs):
                theta = inputs['theta']
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                g = inputs['g']
                v = inputs['v']

                outputs['vdot'] = g * cos_theta
                outputs['xdot'] = v * sin_theta
                outputs['ydot'] = -v * cos_theta
                outputs['check'] = v / sin_theta

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()

        t = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=_BrachODE, transcription=t)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False, rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=False, rate_source='ydot')

        # Intentionally incorrect rate source to trigger an error during configure.
        phase.add_state('v', fix_initial=True, fix_final=False, rate_source='vel_dot')

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2')

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        with self.assertRaises(RuntimeError) as ctx:
            p.setup()

        expected = 'Error during configure_states_introspection in phase traj0.phases.phase0.'
        self.assertEqual(str(ctx.exception), expected)


if __name__ == '__main__':
    unittest.main()
