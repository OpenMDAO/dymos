import os
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestDoubleIntegratorForDocs(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @save_for_docs
    def test_double_integrator_for_docs(self):
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE

        # Initialize the problem and assign the driver
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        # Setup the trajectory and its phase
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=30, order=3, compressed=False)

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=DoubleIntegratorODE, transcription=transcription))

        #
        # Set the options for our variables.
        #
        phase.set_time_options(fix_initial=True, fix_duration=True, units='s')
        phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')
        phase.add_state('x', fix_initial=True, rate_source='v', units='m')

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, shape=(1, ), lower=-1.0, upper=1.0)

        #
        # Maximize distance travelled.
        #
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup the problem and set our initial values.
        #
        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 1.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['traj.phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        #
        # Solve the problem.
        #
        dm.run_problem(p)

        #
        # Verify that the results are correct.
        #
        x = p.get_val('traj.phase0.timeseries.states:x')
        v = p.get_val('traj.phase0.timeseries.states:v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

        #
        # Simulate the explicit solution and plot the results.
        #
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:x',
                       'time (s)', 'x $(m)$'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:v',
                       'time (s)', 'v $(m/s)$'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:u',
                       'time (s)', 'u $(m/s^2)$')],
                     title='Double Integrator Solution\nRadau Pseudospectral Method',
                     p_sol=p, p_sim=exp_out)

        plt.show()


if __name__ == "__main__":
    unittest.main()
