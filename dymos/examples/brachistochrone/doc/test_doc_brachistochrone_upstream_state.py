import unittest
from dymos.utils.doc_utils import save_for_docs


class TestBrachistochroneUpstreamState(unittest.TestCase):

    @save_for_docs
    def test_brachistochrone_upstream_state(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')

        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        # Instantiate the transcription so we can get the number of nodes from it while
        # building the problem.
        tx = dm.GaussLobatto(num_segments=10, order=3)

        # Add an indep var comp to provide the external control values
        ivc = p.model.add_subsystem('states_ivc', om.IndepVarComp(), promotes_outputs=['*'])

        # Add the output to provide the values of theta at the control input nodes of the transcription.
        ivc.add_output('x0', shape=(1,), units='m')

        # Connect x0 to the state error component so we can constrain the given value of x0
        # to be equal to the value chosen in the phase.
        p.model.connect('x0', 'state_error_comp.x0_target')
        p.model.connect('traj.phase0.timeseries.states:x', 'state_error_comp.x0_actual', src_indices=[0])

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        p.model.add_subsystem('state_error_comp',
                              om.ExecComp('x0_error = x0_target - x0_actual',
                                          x0_error={'units': 'm'},
                                          x0_target={'units': 'm'},
                                          x0_actual={'units': 'm'}))

        p.model.add_constraint('state_error_comp.x0_error', equals=0.0)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=tx)

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=False, fix_final=True, units='m', rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot', targets=['v'])

        # Define theta as a control.
        # Use opt=False to allow it to be connected to an external source.
        # Arguments lower and upper are no longer valid for an input control.
        phase.add_control(name='theta', units='rad', targets=['theta'])

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
        p.set_val('x0', 0.0, units='m')

        # Here we're intentially setting the intiial x value to something other than zero, just
        # to demonstrate that the optimizer brings it back in line with the value of x0 set above.
        p.set_val('traj.phase0.states:x',
                  phase.interpolate(ys=[1, 10], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:y',
                  phase.interpolate(ys=[10, 5], nodes='state_input'),
                  units='m')

        p.set_val('traj.phase0.states:v',
                  phase.interpolate(ys=[0, 5], nodes='state_input'),
                  units='m/s')

        p.set_val('traj.phase0.controls:theta',
                  phase.interpolate(ys=[90, 90], nodes='control_input'),
                  units='deg')

        # Run the driver to solve the problem
        dm.run_problem(p, make_plots=True)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
                          tolerance=1.0E-3)

        # Check the validity of our results by using scipy.integrate.solve_ivp to
        # integrate the solution.
        sim_out = traj.simulate()

        # Plot the results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

        axes[0].plot(p.get_val('traj.phase0.timeseries.states:x'),
                     p.get_val('traj.phase0.timeseries.states:y'),
                     'ro', label='solution')

        axes[0].plot(sim_out.get_val('traj.phase0.timeseries.states:x'),
                     sim_out.get_val('traj.phase0.timeseries.states:y'),
                     'b-', label='simulation')

        axes[0].set_xlabel('x (m)')
        axes[0].set_ylabel('y (m/s)')
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
                     p.get_val('traj.phase0.timeseries.controls:theta', units='deg'),
                     'ro', label='solution')

        axes[1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                     sim_out.get_val('traj.phase0.timeseries.controls:theta', units='deg'),
                     'b-', label='simulation')

        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel(r'$\theta$ (deg)')
        axes[1].legend()
        axes[1].grid()

        plt.show()
