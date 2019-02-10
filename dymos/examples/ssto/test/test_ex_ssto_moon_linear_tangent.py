from __future__ import print_function, absolute_import, division

import itertools
import unittest

from numpy.testing import assert_almost_equal

import matplotlib
matplotlib.use('Agg')

from parameterized import parameterized

from openmdao.utils.assert_utils import assert_rel_error

import dymos.examples.ssto.ex_ssto_moon_linear_tangent as ex_ssto_moon_lintan


class TestExampleSSTOMoonLinearTangent(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['fwd'],  # derivative_mode
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1]])
    )
    def test_results(self, transcription='gauss-lobatto', derivative_mode='rev', compressed=True):
        p = ex_ssto_moon_lintan.ssto_moon_linear_tangent(transcription, num_seg=10,
                                                         transcription_order=5,
                                                         compressed=compressed)

        # Ensure defects are zero
        for state in ['x', 'y', 'vx', 'vy', 'm']:
            assert_almost_equal(p['phase0.collocation_constraint.defects:{0}'.format(state)],
                                0.0, decimal=5)

            if not compressed:
                assert_almost_equal(p['phase0.continuity_comp.'
                                      'defect_states:{0}'.format(state)],
                                    0.0, decimal=5,
                                    err_msg='error in state continuity for state {0}'.format(state))

        # Ensure time found is the known solution
        assert_almost_equal(p['phase0.t_duration'], 481.8, decimal=1)

        # Does this case find the same answer as using theta as a dynamic control?
        assert_almost_equal(p['phase0.design_parameters:a_ctrl'], -0.0082805, decimal=4)
        assert_almost_equal(p['phase0.design_parameters:b_ctrl'], 2.74740137, decimal=4)

    def test_plot(self):
        import matplotlib.pyplot as plt
        import dymos.examples.ssto.ex_ssto_moon_linear_tangent as ex_ssto_moon_lintan

        p = ex_ssto_moon_lintan.ssto_moon_linear_tangent(transcription='gauss-lobatto',
                                                         optimizer='SLSQP')

        # quick check of the results
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 1.85E5, 1e-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:vx')[-1], 1627.0, 1e-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:vy')[-1], 0, 1e-4)

        ##############################
        # Plot the trajectory
        ##############################
        plt.figure(facecolor='white')
        plt.plot(p.get_val('phase0.timeseries.states:x'),
                 p.get_val('phase0.timeseries.states:y'),
                 'bo')
        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.grid()

        fig = plt.figure(facecolor='white')
        fig.suptitle('results for flat_earth_without_aero')

        axarr = fig.add_subplot(2, 1, 1)
        axarr.plot(p.get_val('phase0.timeseries.time'),
                   p.get_val('phase0.timeseries.theta', units='deg'), 'bo')
        axarr.set_ylabel(r'$\theta$, deg')
        axarr.axes.get_xaxis().set_visible(False)

        axarr = fig.add_subplot(2, 1, 2)

        axarr.plot(p.get_val('phase0.timeseries.time'),
                   p.get_val('phase0.timeseries.states:vx'), 'bo', label='$v_x$')
        axarr.plot(p.get_val('phase0.timeseries.time'),
                   p.get_val('phase0.timeseries.states:vy'), 'ro', label='$v_y$')
        axarr.set_xlabel('time, s')
        axarr.set_ylabel('velocity, m/s')
        axarr.legend(loc='best')
        plt.show()

if __name__ == "__main__":
    unittest.main()
