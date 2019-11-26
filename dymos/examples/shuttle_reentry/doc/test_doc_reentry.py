from __future__ import print_function, absolute_import, division
import os
import unittest
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.style.use('ggplot')
import numpy as np


class TestReentryForDocs(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_reentry(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_rel_error
        import dymos as dm
        from dymos.examples.shuttle_reentry.shuttle_ode import ShuttleODE
        from dymos.examples.plotting import plot_results

        # Instantiate the problem, add the driver, and allow it to use coloring
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        # Instantiate the trajectory and add a phase to it
        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0',
                                dm.Phase(ode_class=ShuttleODE,
                                         transcription=dm.Radau(num_segments=15, order=3)))

        phase0.set_time_options(fix_initial=True, units='s', duration_ref=200)
        phase0.add_state('h', fix_initial=True, fix_final=True, units='ft', rate_source='hdot',
                         targets=['h'], lower=0, ref0=75000, ref=300000, defect_ref=1000)
        phase0.add_state('gamma', fix_initial=True, fix_final=True, units='rad',
                         rate_source='gammadot', targets=['gamma'],
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('phi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='phidot', lower=0, upper=89. * np.pi / 180)
        phase0.add_state('psi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='psidot', targets=['psi'], lower=0, upper=90. * np.pi / 180)
        phase0.add_state('theta', fix_initial=True, fix_final=False, units='rad',
                         rate_source='thetadot', targets=['theta'],
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('v', fix_initial=True, fix_final=True, units='ft/s',
                         rate_source='vdot', targets=['v'], lower=0, ref0=2500, ref=25000)
        phase0.add_control('alpha', units='rad', opt=True,
                           lower=-np.pi / 2, upper=np.pi / 2, targets=['alpha'])
        phase0.add_control('beta', units='rad', opt=True,
                           lower=-89 * np.pi / 180, upper=1 * np.pi / 180, targets=['beta'])

        # The original implementation by Betts includes a heating rate path constraint.
        # This will work with the SNOPT optimizer but SLSQP has difficulty converging the solution.
        # phase0.add_path_constraint('q', lower=0, upper=70, units='Btu/ft**2/s', ref=70)
        phase0.add_timeseries_output('q', shape=(1,), units='Btu/ft**2/s')

        phase0.add_objective('theta', loc='final', ref=-0.01)

        p.setup(check=True)

        p.set_val('traj.phase0.states:h',
                  phase0.interpolate(ys=[260000, 80000], nodes='state_input'), units='ft')
        p.set_val('traj.phase0.states:gamma',
                  phase0.interpolate(ys=[-1, -5], nodes='state_input'), units='deg')
        p.set_val('traj.phase0.states:phi',
                  phase0.interpolate(ys=[0, 75], nodes='state_input'), units='deg')
        p.set_val('traj.phase0.states:psi',
                  phase0.interpolate(ys=[90, 10], nodes='state_input'), units='deg')
        p.set_val('traj.phase0.states:theta',
                  phase0.interpolate(ys=[0, 25], nodes='state_input'), units='deg')
        p.set_val('traj.phase0.states:v',
                  phase0.interpolate(ys=[25600, 2500], nodes='state_input'), units='ft/s')
        p.set_val('traj.phase0.t_initial', 0, units='s')
        p.set_val('traj.phase0.t_duration', 2000, units='s')
        p.set_val('traj.phase0.controls:alpha',
                  phase0.interpolate(ys=[17.4, 17.4], nodes='control_input'), units='deg')
        p.set_val('traj.phase0.controls:beta',
                  phase0.interpolate(ys=[-75, 0], nodes='control_input'), units='deg')

        # Run the driver
        p.run_driver()

        # Check the validity of the solution
        assert_rel_error(self, p.get_val('traj.phase0.timeseries.time')[-1], 2008.59,
                         tolerance=1e-3)
        assert_rel_error(self, p.get_val('traj.phase0.timeseries.states:theta', units='deg')[-1],
                         34.1412, tolerance=1e-3)
        # assert_rel_error(self, p.get_val('traj.phase0.timeseries.time')[-1], 2181.90371131, tolerance=1e-3)
        # assert_rel_error(self, p.get_val('traj.phase0.timeseries.states:theta')[-1], .53440626, tolerance=1e-3)

        # Run the simulation to check if the model is physically valid
        sim_out = traj.simulate()

        # Plot the results

        plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:alpha',
                       'time (s)', 'alpha (rad)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:beta',
                       'time (s)', 'beta (rad)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:theta',
                       'time (s)', 'theta (rad)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.q',
                       'time (s)', 'q (btu/ft/ft/s')], title='Reentry Solution', p_sol=p,
                     p_sim=sim_out)

        plt.show()


if __name__ == '__main__':
    unittest.main()
