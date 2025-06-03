import os
import unittest

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    plt.style.use('ggplot')
except ImportError:
    matplotlib = None


from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestBrachistochroneStaticGravity(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_partials(self):
        import numpy as np
        import openmdao.api as om
        from dymos.utils.testing_utils import assert_check_partials
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        num_nodes = 5

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('vars', om.IndepVarComp())
        ivc.add_output('v', shape=(num_nodes,), units='m/s')
        ivc.add_output('theta', shape=(num_nodes,), units='deg')

        p.model.add_subsystem('ode', BrachistochroneODE(num_nodes=num_nodes, static_gravity=True))

        p.model.connect('vars.v', 'ode.v')
        p.model.connect('vars.theta', 'ode.theta')

        p.setup(force_alloc_complex=True)

        p.set_val('vars.v', 10*np.random.random(num_nodes))
        p.set_val('vars.theta', 10*np.random.uniform(1, 179, num_nodes))

        p.run_model()
        cpd = p.check_partials(method='cs', compact_print=True)
        assert_check_partials(cpd)

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_static_gravity(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        import matplotlib.pyplot as plt
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

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
                                        ode_init_kwargs={'static_gravity': True},
                                        transcription=dm.GaussLobatto(num_segments=10)))

        #
        # Set the variables
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        targets=None,
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        targets=None,
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        targets=['v'],
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', targets=['theta'],
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], static_target=True, opt=False)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        # The initial time is fixed, and we set that fixed value here.
        # The optimizer is allowed to modify t_duration, but an initial guess is provided here.
        #
        phase.set_time_val(initial=0, duration=2)

        # Guesses for states are provided at all state_input nodes.
        # We use the phase.interpolate method to linearly interpolate values onto the state input nodes.
        # Since fix_initial=True for all states and fix_final=True for x and y, the initial or final
        # values of the interpolation provided here will not be changed by the optimizer.
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        # Guesses for controls are provided at all control_input node.
        # Here phase.interpolate is used to linearly interpolate values onto the control input nodes.
        phase.set_control_val('theta', [5, 100.5])

        # Set the value for gravitational acceleration.
        phase.set_parameter_val('g', 9.80665)

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, simulate=True)

        # Test the results
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016,
                          tolerance=1.0E-3)

        # Load the explicitly simulated trajectory
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        exp_out = om.CaseReader(sim_db).get_case('final')

        # Extract the timeseries from the implicit solution and the explicit simulation
        x = p.get_val('traj.phase0.timeseries.x')
        y = p.get_val('traj.phase0.timeseries.y')
        t = p.get_val('traj.phase0.timeseries.time')
        theta = p.get_val('traj.phase0.timeseries.theta')

        x_exp = exp_out.get_val('traj.phase0.timeseries.x')
        y_exp = exp_out.get_val('traj.phase0.timeseries.y')
        t_exp = exp_out.get_val('traj.phase0.timeseries.time')
        theta_exp = exp_out.get_val('traj.phase0.timeseries.theta')

        fig, axes = plt.subplots(nrows=2, ncols=1)

        axes[0].plot(x, y, 'o')
        axes[0].plot(x_exp, y_exp, '-')
        axes[0].set_xlabel('x (m)')
        axes[0].set_ylabel('y (m)')

        axes[1].plot(t, theta, 'o')
        axes[1].plot(t_exp, theta_exp, '-')
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel(r'$\theta$ (deg)')

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
