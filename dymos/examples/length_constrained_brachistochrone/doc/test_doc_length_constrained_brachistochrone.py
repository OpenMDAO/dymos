from __future__ import print_function, division, absolute_import

import unittest
from dymos.utils.doc_utils import save_for_docs

import matplotlib
matplotlib.use('Agg')


SHOW_PLOTS = True


class TestLengthConstrainedBrachistochrone(unittest.TestCase):

    @save_for_docs
    def test_length_constrained_brachistochrone(self):
        import openmdao.api as om
        import dymos as dm
        import matplotlib.pyplot as plt
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
        from dymos.examples.length_constrained_brachistochrone.arc_length_comp import ArcLengthComp

        MAX_ARCLENGTH = 11.9
        OPTIMIZER = 'SLSQP'

        p = om.Problem(model=om.Group())
        p.add_recorder(om.SqliteRecorder('length_constrained_brach_sol.db'))

        if OPTIMIZER == 'SNOPT':
            p.driver = om.pyOptSparseDriver()
            p.driver.options['optimizer'] = OPTIMIZER
            p.driver.opt_settings['Major iterations limit'] = 1000
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = om.ScipyOptimizeDriver()

        p.driver.declare_coloring()

        # Create the transcription so we can get the number of nodes for the downstream analysis
        tx = dm.Radau(num_segments=20, order=3, compressed=False)

        traj = dm.Trajectory()
        phase = dm.Phase(transcription=tx, ode_class=BrachistochroneODE)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', units='m', rate_source='xdot', fix_initial=True, fix_final=True)
        phase.add_state('y', units='m', rate_source='ydot', fix_initial=True, fix_final=True)
        phase.add_state('v', units='m/s', rate_source='vdot', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9,
                          continuity=True, rate_continuity=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=1)

        # p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
        # p.model.linear_solver = DirectSolver(assemble_jac=True)

        # Add the arc length component
        p.model.add_subsystem('arc_length_comp',
                              subsys=ArcLengthComp(num_nodes=tx.grid_data.num_nodes))

        p.model.connect('traj.phase0.timeseries.controls:theta', 'arc_length_comp.theta')
        p.model.connect('traj.phase0.timeseries.states:x', 'arc_length_comp.x')

        p.model.add_constraint('arc_length_comp.S', upper=MAX_ARCLENGTH, ref=1)

        p.setup(check=True)

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase.interpolate(ys=[0, 10], nodes='state_input'))
        p.set_val('traj.phase0.states:y', phase.interpolate(ys=[10, 5], nodes='state_input'))
        p.set_val('traj.phase0.states:v', phase.interpolate(ys=[0, 9.9], nodes='state_input'))
        p.set_val('traj.phase0.controls:theta', phase.interpolate(ys=[5, 100], nodes='control_input'))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        p.run_driver()

        p.record(case_name='final')

        # Plot results
        if SHOW_PLOTS:
            # Generate the explicitly simulated trajectory
            exp_out = traj.simulate()

            # Extract the timeseries from the implicit solution and the explicit simulation
            x = p.get_val('traj.phase0.timeseries.states:x')
            y = p.get_val('traj.phase0.timeseries.states:y')
            t = p.get_val('traj.phase0.timeseries.time')
            theta = p.get_val('traj.phase0.timeseries.controls:theta')

            x_exp = exp_out.get_val('traj.phase0.timeseries.states:x')
            y_exp = exp_out.get_val('traj.phase0.timeseries.states:y')
            t_exp = exp_out.get_val('traj.phase0.timeseries.time')
            theta_exp = exp_out.get_val('traj.phase0.timeseries.controls:theta')

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

        return p


if __name__ == '__main__':
    unittest.main()
