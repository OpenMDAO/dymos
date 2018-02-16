from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdoc.examples.min_time_climb.min_time_climb_problem import min_time_climb_problem

SHOW_PLOTS = False


class TestMinTimeClimb_GaussLobatto(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = min_time_climb_problem(optimizer='SLSQP', num_seg=10,
                                       transcription_order=[5, 3, 5, 3, 5, 3, 5, 3, 5, 3],
                                       transcription='gauss-lobatto', top_level_densejacobian=True)

        cls.p.run_driver()

        cls.phase = cls.p.model.phase0

        # Check that time matches to within 1% of an externally verified solution.
        assert_almost_equal((cls.phase.get_values('time')[-1]-320.0)/320.0, 0.0, decimal=2)

        cls.exp_out = cls.phase.simulate(times=np.linspace(0, cls.p['phase0.t_duration'], 100))

    def test_get_values(self):
        gd = self.phase.grid_data

        idx_disc = gd.subset_node_indices['disc']
        idx_col = gd.subset_node_indices['col']

        for var in ['time', 'r', 'h', 'v', 'gam', 'alpha', 'alpha_rate', 'alpha_rate2']:
            col_vals = self.phase.get_values(var=var, nodes='col')
            disc_vals = self.phase.get_values(var=var, nodes='disc')
            all_vals = self.phase.get_values(var=var, nodes='all')

            assert_almost_equal(col_vals, all_vals[idx_col])
            assert_almost_equal(disc_vals, all_vals[idx_disc])

    def test_show_plots(self):
        if not SHOW_PLOTS:
            self.skipTest('SHOW_PLOTS is False, skipping plot generation.')

        import matplotlib.pyplot as plt
        plt.plot(self.phase.get_values('time'), self.phase.get_values('h'), 'ro')
        plt.plot(self.exp_out.get_values('time'), self.exp_out.get_values('h'), 'b-')
        plt.xlabel('time (s)')
        plt.ylabel('altitude (m)')

        plt.figure()
        plt.plot(self.phase.get_values('v'), self.phase.get_values('h'), 'ro')
        plt.plot(self.exp_out.get_values('v'), self.exp_out.get_values('h'), 'b-')
        plt.xlabel('airspeed (m/s)')
        plt.ylabel('altitude (m)')

        plt.show()


class TestMinTimeClimb_Radau(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = min_time_climb_problem(optimizer='SLSQP', num_seg=10,
                                       transcription_order=[5, 3, 5, 3, 5, 3, 5, 3, 5, 3],
                                       transcription='radau-ps', top_level_densejacobian=True)

        cls.p.run_driver()

        cls.phase = cls.p.model.phase0

        # Check that time matches to within 1% of an externally verified solution.
        assert_almost_equal((cls.phase.get_values('time')[-1]-320.0)/320.0, 0.0, decimal=2)

        cls.exp_out = cls.phase.simulate(times=np.linspace(0, cls.p['phase0.t_duration'], 100))

    def test_get_values(self):
        gd = self.phase.grid_data

        idx_disc = gd.subset_node_indices['disc']
        idx_col = gd.subset_node_indices['col']

        for var in ['time', 'r', 'h', 'v', 'gam', 'alpha', 'alpha_rate', 'alpha_rate2']:
            col_vals = self.phase.get_values(var=var, nodes='col')
            disc_vals = self.phase.get_values(var=var, nodes='disc')
            all_vals = self.phase.get_values(var=var, nodes='all')

            assert_almost_equal(col_vals, all_vals[idx_col])
            assert_almost_equal(disc_vals, all_vals[idx_disc])

    def test_show_plots(self):
        if not SHOW_PLOTS:
            self.skipTest('SHOW_PLOTS is False, skipping plot generation.')

        import matplotlib.pyplot as plt
        plt.plot(self.phase.get_values('time'), self.phase.get_values('h'), 'ro')
        plt.plot(self.exp_out.get_values('time'), self.exp_out.get_values('h'), 'b-')
        plt.xlabel('time (s)')
        plt.ylabel('altitude (m)')

        plt.figure()
        plt.plot(self.phase.get_values('v'), self.phase.get_values('h'), 'ro')
        plt.plot(self.exp_out.get_values('v'), self.exp_out.get_values('h'), 'b-')
        plt.xlabel('airspeed (m/s)')
        plt.ylabel('altitude (m)')

        plt.show()


if __name__ == '__main__':
    unittest.main()
