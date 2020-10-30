import unittest

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.style.use('ggplot')

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from dymos.utils.doc_utils import save_for_docs


@use_tempdirs
class TestDocSSTOEarth(unittest.TestCase):

    @save_for_docs
    def test_doc_ssto_earth(self):
        import matplotlib.pyplot as plt
        import openmdao.api as om
        import dymos as dm

        #
        # Setup and solve the optimal control problem
        #
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE

        #
        # Initialize our Trajectory and Phase
        #
        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=LaunchVehicleODE,
                         ode_init_kwargs={'central_body': 'earth'},
                         transcription=dm.GaussLobatto(num_segments=12, order=3, compressed=False))

        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        #
        # Set the options for the variables
        #
        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 500))

        phase.add_state('x', fix_initial=True, ref=1.0E5, defect_ref=1.0,
                        rate_source='eom.xdot', units='m')
        phase.add_state('y', fix_initial=True, ref=1.0E5, defect_ref=1.0,
                        rate_source='eom.ydot', targets=['atmos.y'], units='m')
        phase.add_state('vx', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                        rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
        phase.add_state('vy', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                        rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
        phase.add_state('m', fix_initial=True, ref=1.0E3, defect_ref=1.0,
                        rate_source='eom.mdot', targets=['eom.m'], units='kg')

        phase.add_control('theta', units='rad', lower=-1.57, upper=1.57, targets=['eom.theta'])
        phase.add_parameter('thrust', units='N', opt=False, val=2100000.0, targets=['eom.thrust'])

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

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 150.0)
        p.set_val('traj.phase0.states:x', phase.interpolate(ys=[0, 1.15E5], nodes='state_input'))
        p.set_val('traj.phase0.states:y', phase.interpolate(ys=[0, 1.85E5], nodes='state_input'))
        p.set_val('traj.phase0.states:vx', phase.interpolate(ys=[0, 7796.6961], nodes='state_input'))
        p.set_val('traj.phase0.states:vy', phase.interpolate(ys=[1.0E-6, 0], nodes='state_input'))
        p.set_val('traj.phase0.states:m', phase.interpolate(ys=[117000, 1163], nodes='state_input'))
        p.set_val('traj.phase0.controls:theta', phase.interpolate(ys=[1.5, -0.76], nodes='control_input'))
        p.set_val('traj.phase0.parameters:thrust', 2.1, units='MN')

        #
        # Solve the Problem
        #
        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 143, tolerance=0.05)
        assert_near_equal(p.get_val('traj.phase0.timeseries.states:y')[-1], 1.85E5, 1e-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.states:vx')[-1], 7796.6961, 1e-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.states:vy')[-1], 0, 1e-4)
        #
        # Get the explicitly simulated results
        #
        exp_out = traj.simulate()

        #
        # Plot the results
        #
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        axes[0].plot(p.get_val('traj.phase0.timeseries.states:x'),
                     p.get_val('traj.phase0.timeseries.states:y'),
                     marker='o',
                     ms=4,
                     linestyle='None',
                     label='solution')

        axes[0].plot(exp_out.get_val('traj.phase0.timeseries.states:x'),
                     exp_out.get_val('traj.phase0.timeseries.states:y'),
                     marker=None,
                     linestyle='-',
                     label='simulation')

        axes[0].set_xlabel('range (m)')
        axes[0].set_ylabel('altitude (m)')
        axes[0].set_aspect('equal')

        axes[1].plot(p.get_val('traj.phase0.timeseries.time'),
                     p.get_val('traj.phase0.timeseries.controls:theta'),
                     marker='o',
                     ms=4,
                     linestyle='None')

        axes[1].plot(exp_out.get_val('traj.phase0.timeseries.time'),
                     exp_out.get_val('traj.phase0.timeseries.controls:theta'),
                     linestyle='-',
                     marker=None)

        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('theta (deg)')

        plt.suptitle('Single Stage to Orbit Solution Using Linear Tangent Guidance')
        fig.legend(loc='lower center', ncol=2)

        plt.show()


if __name__ == "__main__":
    unittest.main()
