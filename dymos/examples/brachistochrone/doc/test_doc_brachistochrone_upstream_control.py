import unittest

from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.doc_utils import save_for_docs


@use_tempdirs
class TestBrachistochroneUpstreamControl(unittest.TestCase):

    @save_for_docs
    def test_brachistochrone_upstream_control(self):
        import numpy as np
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
        ivc = p.model.add_subsystem('control_ivc', om.IndepVarComp(), promotes_outputs=['*'])

        # Add the output to provide the values of theta at the control input nodes of the transcription.
        ivc.add_output('theta', shape=(tx.grid_data.subset_num_nodes['control_input']), units='rad')

        # Add this external control as a design variable
        p.model.add_design_var('theta', units='rad', lower=1.0E-5, upper=np.pi)
        # Connect this to controls:theta in the appropriate phase.
        # connect calls are cached, so we can do this before we actually add the trajectory to the problem.
        p.model.connect('theta', 'traj.phase0.controls:theta')

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

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
        phase.add_state('x', fix_initial=True, fix_final=True, units='m', rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot', targets=['v'])

        # Define theta as a control.
        # Use opt=False to allow it to be connected to an external source.
        # Arguments lower and upper are no longer valid for an input control.
        phase.add_control(name='theta', targets=['theta'], opt=False)

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states and controls.
        p.set_val('traj.phase0.states:x', phase.interp('x', [0, 10]), units='m')

        p.set_val('traj.phase0.states:y', phase.interp('y', [10, 5]), units='m')

        p.set_val('traj.phase0.states:v', phase.interp('v', [0, 5]), units='m/s')

        p.set_val('traj.phase0.controls:theta', phase.interp('theta', [90, 90]), units='deg')

        # Run the driver to solve the problem
        p.run_driver()

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
