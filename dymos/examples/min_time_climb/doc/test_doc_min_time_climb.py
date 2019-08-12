from __future__ import print_function, absolute_import, division

import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class TestMinTimeClimbForDocs(unittest.TestCase):

    def test_min_time_climb_for_docs_gauss_lobatto(self):
        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_rel_error

        import dymos as dm
        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
        from dymos.examples.plotting import plot_results

        #
        # Instantiate the problem and configure the optimization driver
        #
        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        #
        # Instantiate the trajectory and phase
        #
        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                         transcription=dm.GaussLobatto(num_segments=15, compressed=True))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        #
        # Set the options on the optimization variables
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                        ref=1.0E3, defect_ref=1.0E3, units='m',
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m',
                        rate_source='flight_dynamics.h_dot', targets=['h'])

        phase.add_state('v', fix_initial=True, lower=10.0,
                        ref=1.0E2, defect_ref=1.0E2, units='m/s',
                        rate_source='flight_dynamics.v_dot', targets=['v'])

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5,
                        ref=1.0, defect_ref=1.0, units='rad',
                        rate_source='flight_dynamics.gam_dot', targets=['gam'])

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                        ref=1.0E3, defect_ref=1.0E3, units='kg',
                        rate_source='prop.m_dot', targets=['m'])

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False, targets=['alpha'])

        phase.add_design_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
        phase.add_design_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
        phase.add_design_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        #
        # Setup the boundary and path constraints
        #
        phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, shape=(1,))
        phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

        phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8, shape=(1,))

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', ref=1.0)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 500

        p['traj.phase0.states:r'] = phase.interpolate(ys=[0.0, 50000.0], nodes='state_input')
        p['traj.phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
        p['traj.phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
        p['traj.phase0.states:m'] = phase.interpolate(ys=[19030.468, 10000.], nodes='state_input')
        p['traj.phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

        #
        # Solve for the optimal trajectory
        #
        p.run_driver()

        #
        # Test the results
        #
        assert_rel_error(self, p.get_val('traj.phase0.t_duration'), 321.0, tolerance=1.0E-1)

        #
        # Get the explicitly simulated solution and plot the results
        #
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:h',
                       'time (s)', 'altitude (m)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:alpha',
                       'time (s)', 'alpha (deg)')],
                     title='Supersonic Minimum Time-to-Climb Solution',
                     p_sol=p, p_sim=exp_out)

        plt.show()


if __name__ == '__main__':
    unittest.main()
