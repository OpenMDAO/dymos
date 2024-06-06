import unittest

try:
    import matplotlib
except ImportError:
    matplotlib = None

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


@use_tempdirs
class TestDocSSTOEarth(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_doc_ssto_earth(self):
        import matplotlib.pyplot as plt
        import openmdao.api as om
        import dymos as dm

        #
        # Setup and solve the optimal control problem
        #
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE

        #
        # Initialize our Trajectory and Phase
        #
        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=LaunchVehicleODE,
                         transcription=dm.GaussLobatto(num_segments=12, order=3, compressed=False))

        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        #
        # Set the options for the variables
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(10, 500))

        phase.add_state('x', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                        rate_source='xdot')
        phase.add_state('y', fix_initial=True, ref=1.0E5, defect_ref=10000.0,
                        rate_source='ydot')
        phase.add_state('vx', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                        rate_source='vxdot')
        phase.add_state('vy', fix_initial=True, ref=1.0E3, defect_ref=1000.0,
                        rate_source='vydot')
        phase.add_state('m', fix_initial=True, ref=1.0E3, defect_ref=100.0,
                        rate_source='mdot')

        phase.add_control('theta', units='rad', lower=-1.57, upper=1.57, targets=['theta'])
        phase.add_parameter('thrust', units='N', opt=False, val=2100000.0, targets=['thrust'])

        #
        # Set the options for our constraints and objective
        #
        phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
        phase.add_boundary_constraint('vx', loc='final', equals=7796.6961)
        phase.add_boundary_constraint('vy', loc='final', equals=0)

        phase.add_objective('time', loc='final', scaler=0.01)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup and set initial values
        #
        p.setup(check=True)

        phase.set_time_val(initial=0.0, duration=150.0)
        phase.set_state_val('x', [0, 1.15E5])
        phase.set_state_val('y', [0, 1.85E5])
        phase.set_state_val('vy', [1.0E-6, 0])
        phase.set_state_val('m', [117000, 1163])
        phase.set_control_val('theta', [1.5, -0.76])
        phase.set_parameter_val('thrust', 2.1, units='MN')

        #
        # Solve the Problem
        #
        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 143, tolerance=0.05)
        assert_near_equal(p.get_val('traj.phase0.timeseries.y')[-1], 1.85E5, 1e-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.vx')[-1], 7796.6961, 1e-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.vy')[-1], 0, 1e-4)
        #
        # Get the explicitly simulated results
        #
        exp_out = traj.simulate()

        #
        # Plot the results
        #
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        axes[0].plot(p.get_val('traj.phase0.timeseries.x'),
                     p.get_val('traj.phase0.timeseries.y'),
                     marker='o',
                     ms=4,
                     linestyle='None',
                     label='solution')

        axes[0].plot(exp_out.get_val('traj.phase0.timeseries.x'),
                     exp_out.get_val('traj.phase0.timeseries.y'),
                     marker=None,
                     linestyle='-',
                     label='simulation')

        axes[0].set_xlabel('range (m)')
        axes[0].set_ylabel('altitude (m)')
        axes[0].set_aspect('equal')

        axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
                     p.get_val('traj.phase0.timeseries.theta'),
                     marker='o',
                     ms=4,
                     linestyle='None')

        axes[1].plot(exp_out.get_val('traj.phase0.timeseries.time'),
                     exp_out.get_val('traj.phase0.timeseries.theta'),
                     linestyle='-',
                     marker=None)

        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('theta (deg)')

        plt.suptitle('Single Stage to Orbit Solution Using Linear Tangent Guidance')
        fig.legend(loc='lower center', ncol=2)

        plt.show()


if __name__ == "__main__":
    unittest.main()
