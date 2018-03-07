from __future__ import print_function, absolute_import, division

import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from parameterized import parameterized

import dymos.examples.ssto.ex_ssto_moon as ex_ssto_moon


class TestExampleSSTOMoon(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['csc', 'dense'],  # jacobian
                          ['rev'],  # derivative_mode
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1],
                                                                          p.args[2]])
    )
    def test_results(self, transcription='gauss-lobatto', jacobian='csc', derivative_mode='rev'):
        ex_ssto_moon.SHOW_PLOTS = False
        p = ex_ssto_moon.ssto_moon(transcription, num_seg=10, transcription_order=5,
                                   top_level_jacobian=jacobian, derivative_mode=derivative_mode)

        # Ensure defects are zero
        for state in ['x', 'y', 'vx', 'vy', 'm']:
            assert_almost_equal(p['phase0.collocation_constraint.defects:{0}'.format(state)],
                                0.0, decimal=5)

            assert_almost_equal(p['phase0.continuity_constraint.'
                                  'defect_states:{0}'.format(state)],
                                0.0, decimal=5,
                                err_msg='error in state continuity for state {0}'.format(state))

        # Ensure time found is the known solution
        assert_almost_equal(p['phase0.t_duration'], 481.8, decimal=1)

        # Ensure the tangent of theta is (approximately) linear
        time = p.model.phase0.get_values('time').flatten()
        tan_theta = np.tan(p.model.phase0.get_values('theta').flatten())

        coeffs, residuals, _, _, _ = np.polyfit(time, tan_theta, deg=1, full=True)

        assert_almost_equal(residuals**2, 0.0, decimal=4)
