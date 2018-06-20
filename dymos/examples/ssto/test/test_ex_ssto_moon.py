from __future__ import print_function, absolute_import, division

import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from parameterized import parameterized
from itertools import product

from openmdao.utils.assert_utils import assert_rel_error

import dymos.examples.ssto.ex_ssto_moon as ex_ssto_moon

import matplotlib
matplotlib.use('Agg')


class TestExampleSSTOMoon(unittest.TestCase):

    def run_asserts(self, p, transcription, compressed):
        # Ensure defects are zero
        for state in ['x', 'y', 'vx', 'vy', 'm']:
            assert_almost_equal(p['phase0.collocation_constraint.defects:{0}'.format(state)],
                                0.0, decimal=5)

        # Ensure time found is the known solution
        assert_almost_equal(p['phase0.t_duration'], 481.8, decimal=1)

        # Ensure the tangent of theta is (approximately) linear
        time = p.model.phase0.get_values('time').flatten()
        tan_theta = np.tan(p.model.phase0.get_values('theta').flatten())

        coeffs, residuals, _, _, _ = np.polyfit(time, tan_theta, deg=1, full=True)

        assert_almost_equal(residuals**2, 0.0, decimal=4)

    # @parameterized.expand(
    #     itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
    #                       ['dense', 'csc'],  # jacobian
    #                       ['rev'],  # derivative_mode
    #                       ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
    #                                                                       p.args[0],
    #                                                                       p.args[1],
    #                                                                       p.args[2]])
    # )
    def test_results(self, transcription='gauss-lobatto', jacobian='csc', derivative_mode='fwd',
                     compressed=False):
        p = ex_ssto_moon.ssto_moon(transcription, num_seg=10, transcription_order=5,
                                   top_level_jacobian=jacobian, compressed=compressed)

        p.setup(mode=derivative_mode, check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500.0

        phase = p.model.phase0
        p['phase0.states:x'] = phase.interpolate(ys=[0, 350000.0], nodes='state_disc')
        p['phase0.states:y'] = phase.interpolate(ys=[0, 185000.0], nodes='state_disc')
        p['phase0.states:vx'] = phase.interpolate(ys=[0, 1627.0], nodes='state_disc')
        p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='state_disc')
        p['phase0.states:m'] = phase.interpolate(ys=[50000, 50000], nodes='state_disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[1.5, -0.76], nodes='control_disc')

        p.run_driver()

        self.run_asserts(p, transcription, compressed)

    def test_plot(self):
        import matplotlib.pyplot as plt

        import dymos.examples.ssto.ex_ssto_moon as ex_ssto_moon

        p = ex_ssto_moon.ssto_moon('gauss-lobatto', num_seg=10,
                                   transcription_order=5, top_level_jacobian='csc')

        p.setup(mode='fwd', check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500.0

        phase = p.model.phase0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 350000.0], nodes='state_disc')
        p['phase0.states:y'] = phase.interpolate(ys=[0, 185000.0], nodes='state_disc')
        p['phase0.states:vx'] = phase.interpolate(ys=[0, 1627.0], nodes='state_disc')
        p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='state_disc')
        p['phase0.states:m'] = phase.interpolate(ys=[50000, 50000], nodes='state_disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[1.5, -0.76], nodes='control_disc')

        p.run_driver()

        ##############################
        # quick check of the results
        ##############################
        assert_rel_error(self, phase.get_values('y')[-1], 1.85E5, 1e-4)
        assert_rel_error(self, phase.get_values('vx')[-1], 1627.0, 1e-4)
        assert_rel_error(self, phase.get_values('vy')[-1], 0, 1e-4)

        ##############################
        # Plot the trajectory
        ##############################
        plt.figure(facecolor='white')
        plt.plot(phase.get_values('x'), phase.get_values('y'), 'bo')
        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.grid()

        fig = plt.figure(facecolor='white')
        fig.suptitle('results for flat_earth_without_aero')

        axarr = fig.add_subplot(2, 1, 1)
        axarr.plot(phase.get_values('time'),
                   np.degrees(phase.get_values('theta')), 'bo')
        axarr.set_ylabel(r'$\theta$, deg')
        axarr.axes.get_xaxis().set_visible(False)

        axarr = fig.add_subplot(2, 1, 2)

        axarr.plot(phase.get_values('time'),
                   np.degrees(phase.get_values('vx')), 'bo', label='$v_x$')
        axarr.plot(phase.get_values('time'),
                   np.degrees(phase.get_values('vy')), 'ro', label='$v_y$')
        axarr.set_xlabel('time, s')
        axarr.set_ylabel('velocity, m/s')
        axarr.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    unittest.main()
