from __future__ import print_function, absolute_import, division

import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from parameterized import parameterized

import dymos.examples.ssto.ex_ssto_moon_linear_tangent as ex_ssto_moon_lintan


class TestExampleSSTOMoonLinearTangent(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['csc'],  # jacobian
                          ['rev'],  # derivative_mode
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1],
                                                                          p.args[2]])
    )
    def test_results(self, transcription='gauss-lobatto', jacobian='csc', derivative_mode='rev'):
        ex_ssto_moon_lintan.SHOW_PLOTS = False
        p = ex_ssto_moon_lintan.ssto_moon_linear_tangent(transcription, num_seg=10,
                                                         transcription_order=5,
                                                         top_level_jacobian=jacobian)

        p.setup(mode=derivative_mode, check=True)

        phase = p.model.phase0
        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500.0
        p['phase0.states:x'] = phase.interpolate(ys=[0, 350000.0], nodes='disc')
        p['phase0.states:y'] = phase.interpolate(ys=[0, 185000.0], nodes='disc')
        p['phase0.states:vx'] = phase.interpolate(ys=[0, 1627.0], nodes='disc')
        p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='disc')
        p['phase0.states:m'] = phase.interpolate(ys=[50000, 50000], nodes='disc')
        p['phase0.controls:a_ctrl'] = -0.01
        p['phase0.controls:b_ctrl'] = 3.0

        p.run_driver()

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

        # Does this case find the same answer as using theta as a dynamic control?
        assert_almost_equal(p['phase0.controls:a_ctrl'], -0.0082805, decimal=4)
        assert_almost_equal(p['phase0.controls:b_ctrl'], 2.74740137, decimal=4)


if __name__ == "__main__":
    unittest.main()
