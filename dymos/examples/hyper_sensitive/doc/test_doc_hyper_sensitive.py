from __future__ import print_function, division, absolute_import

import os
import unittest
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


tf = np.float128(10)


def solution():
    sqrt_two = np.sqrt(2)
    val = sqrt_two * tf
    c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
    c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

    ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
    uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
    J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
               (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)
    return ui, uf, J


@use_tempdirs
class TestHyperSensitive(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @save_for_docs
    def test_hyper_sensitive_for_docs(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results
        from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE

        # Initialize the problem and assign the driver
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        # Setup the trajectory and its phase
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=30, order=3, compressed=False)

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=HyperSensitiveODE, transcription=transcription))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
        phase.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
        phase.add_control('u', opt=True, targets=['u'])

        phase.add_boundary_constraint('x', loc='final', equals=1)

        phase.add_objective('xL', loc='final')

        p.setup(check=True)

        p.set_val('traj.phase0.states:x', phase.interp('x', [1.5, 1]))
        p.set_val('traj.phase0.states:xL', phase.interp('xL', [0, 1]))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase.interp('u', [-0.6, 2.4]))

        #
        # Solve the problem.
        #
        dm.run_problem(p)

        #
        # Verify that the results are correct.
        #
        ui, uf, J = solution()

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[0],
                          ui,
                          tolerance=1.5e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[-1],
                          uf,
                          tolerance=1.5e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:xL')[-1],
                          J,
                          tolerance=1e-2)

        #
        # Simulate the explicit solution and plot the results.
        #
        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:x',
                       'time (s)', 'x $(m)$'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:u',
                       'time (s)', 'u $(m/s^2)$')],
                     title='Hyper Sensitive Problem Solution\nRadau Pseudospectral Method',
                     p_sol=p, p_sim=exp_out)

        plt.show()


if __name__ == "__main__":
    unittest.main()
