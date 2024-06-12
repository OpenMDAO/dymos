import os
import unittest

try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    plt.style.use('ggplot')
except ImportError:
    plt = None

import numpy as np

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


@use_tempdirs
class TestReentryForDocs(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @require_pyoptsparse(optimizer='SLSQP')
    @unittest.skipIf(plt is None, "This test requires matplotlib")
    def test_reentry(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
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
                         lower=0, ref0=75000, ref=300000, defect_ref=1000)
        phase0.add_state('gamma', fix_initial=True, fix_final=True, units='rad',
                         rate_source='gammadot',
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('phi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='phidot', lower=0, upper=89. * np.pi / 180)
        phase0.add_state('psi', fix_initial=True, fix_final=False, units='rad',
                         rate_source='psidot', lower=0, upper=90. * np.pi / 180)
        phase0.add_state('theta', fix_initial=True, fix_final=False, units='rad',
                         rate_source='thetadot',
                         lower=-89. * np.pi / 180, upper=89. * np.pi / 180)
        phase0.add_state('v', fix_initial=True, fix_final=True, units='ft/s',
                         rate_source='vdot', lower=0, ref0=2500, ref=25000)
        phase0.add_control('alpha', units='rad', opt=True, lower=-np.pi / 2, upper=np.pi / 2, )
        phase0.add_control('beta', units='rad', opt=True, lower=-89 * np.pi / 180, upper=1 * np.pi / 180, )

        # The original implementation by Betts includes a heating rate path constraint.
        # This will work with the SNOPT optimizer but SLSQP has difficulty converging the solution.
        # phase0.add_path_constraint('q', lower=0, upper=70, ref=70)
        phase0.add_timeseries_output('q', shape=(1,))

        phase0.add_objective('theta', loc='final', ref=-0.01)

        p.setup(check=True)

        phase0.set_time_val(initial=0, duration=2000, units='s')
        phase0.set_state_val('h', [260000, 80000], units='ft')
        phase0.set_state_val('gamma', [-1, -5], units='deg')
        phase0.set_state_val('phi', [0, 75], units='deg')
        phase0.set_state_val('psi', [90, 10], units='deg')
        phase0.set_state_val('theta', [0, 25], units='deg')
        phase0.set_state_val('v', [25600, 2500], units='ft/s')
        phase0.set_control_val('alpha', 17.4, units='deg')
        phase0.set_control_val('beta', [-75, 0], units='deg')

        # Run the driver
        dm.run_problem(p)

        # Check the validity of the solution
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 2008.59,
                          tolerance=1e-3)
        assert_near_equal(p.get_val('traj.phase0.timeseries.theta', units='deg')[-1],
                          34.1412, tolerance=1e-3)

        # Run the simulation to check if the model is physically valid
        sim_out = traj.simulate()

        # Plot the results

        plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.alpha',
                       'time (s)', 'alpha (rad)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.beta',
                       'time (s)', 'beta (rad)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.theta',
                       'time (s)', 'theta (rad)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.q',
                       'time (s)', 'q (btu/ft/ft/s')], title='Reentry Solution', p_sol=p,
                     p_sim=sim_out)

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
