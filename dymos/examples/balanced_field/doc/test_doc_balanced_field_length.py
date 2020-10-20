import unittest

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs


class TestBalancedFieldLengthForDocs(unittest.TestCase):

    @save_for_docs
    def test_balanced_field_lenth_for_docs(self):
        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.balanced_field.ground_roll_ode import GroundRollODE
        from dymos.examples.plotting import plot_results

        #
        # Instantiate the problem and configure the optimization driver
        #
        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['print_level'] = 5
        p.driver.declare_coloring()

        #
        # Instantiate the trajectory and phase
        #
        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=GroundRollODE,
                         transcription=dm.Radau(num_segments=15))

        traj.add_phase('break_release_to_engine_failure', phase)

        p.model.add_subsystem('traj', traj)

        #
        # Set the options on the optimization variables
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(5, 60),
                               duration_ref=10.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='dynamics.r_dot')

        phase.add_state('v', fix_initial=True, lower=10.0,
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='dynamics.v_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='prop.m_dot')

        phase.add_parameter('h', opt=False, units='m')
        phase.add_parameter('alpha', opt=False, units='deg')

        phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
        phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        phase.add_timeseries_output('*')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', ref=1.0)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p['traj.break_release_to_engine_failure.t_initial'] = 0.0
        p['traj.break_release_to_engine_failure.t_duration'] = 30

        p['traj.break_release_to_engine_failure.states:r'] = phase.interpolate(ys=[0.0, 5000.0], nodes='state_input')
        p['traj.break_release_to_engine_failure.states:v'] = phase.interpolate(ys=[0, 100.0], nodes='state_input')
        p['traj.break_release_to_engine_failure.states:m'] = phase.interpolate(ys=[19030.468, 18000.], nodes='state_input')

        p.set_val('traj.break_release_to_engine_failure.parameters:alpha', 10, units='deg')
        p.set_val('traj.break_release_to_engine_failure.parameters:h', 0.0)

        #
        # Solve for the optimal trajectory
        #
        p.run_model()
        #
        # #
        # # Test the results
        # #
        # assert_near_equal(p.get_val('traj.phase0.t_duration'), 321.0, tolerance=1.0E-1)

        #
        # Get the explicitly simulated solution and plot the results
        #
        exp_out = traj.simulate()

        plt.switch_backend('TkAgg')

        fig, axes = plt.subplots(4, 1)

        axes[0].plot(exp_out.get_val('traj.break_release_to_engine_failure.timeseries.time'),
                     exp_out.get_val('traj.break_release_to_engine_failure.timeseries.states:r', units='ft'))

        axes[1].plot(exp_out.get_val('traj.break_release_to_engine_failure.timeseries.time'),
                     exp_out.get_val('traj.break_release_to_engine_failure.timeseries.states:v', units='kn'))

        axes[2].plot(exp_out.get_val('traj.break_release_to_engine_failure.timeseries.time'),
                     exp_out.get_val('traj.break_release_to_engine_failure.timeseries.F_r', units='lbf'))

        axes[3].plot(exp_out.get_val('traj.break_release_to_engine_failure.timeseries.time'),
                     exp_out.get_val('traj.break_release_to_engine_failure.timeseries.f_lift', units='lbf'))

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
