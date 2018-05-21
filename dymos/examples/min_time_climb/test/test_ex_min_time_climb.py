from __future__ import absolute_import, division, print_function

import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from parameterized import parameterized

from openmdao.utils.assert_utils import assert_rel_error
import dymos.examples.min_time_climb.ex_min_time_climb as ex_min_time_climb

SHOW_PLOTS = True


class TestExampleMinTimeClimb(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['csc'],  # jacobian
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1]])
    )
    def test_results(self, transcription='gauss-lobatto', jacobian='csc'):
        ex_min_time_climb.SHOW_PLOTS = True
        p = ex_min_time_climb.min_time_climb(optimizer='SLSQP',
                                             num_seg=12,
                                             transcription_order=3,
                                             transcription=transcription,
                                             top_level_jacobian=jacobian)

        phase = p.model.phase0

        # Check that time matches to within 1% of an externally verified solution.
        assert_almost_equal((phase.get_values('time')[-1] - 321.0) / 321.0, 0.0, decimal=2)

    # @parameterized.expand(itertools.product(
    #     ['optimizer-based'],
    #     ['GaussLegendre2'],
    # ))
    @unittest.skip('Pruning GLM examples for now')
    def test_results_glm(self, formulation='optimizer-based', method_name='GaussLegendre2'):
        ex_min_time_climb.SHOW_PLOTS = False
        p = ex_min_time_climb.min_time_climb(
            optimizer='SLSQP', num_seg=10, transcription='glm',
            formulation=formulation, method_name=method_name, force_alloc_complex=False)

        phase = p.model.phase0

        # Check that time matches to within 1% of an externally verified solution.
        solution = 321.0
        print(phase.get_values('time'))
        assert_rel_error(self, phase.get_values('time')[-1], solution, tolerance=0.01)

if __name__ == '__main__':
    unittest.main()
