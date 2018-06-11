from __future__ import absolute_import, division, print_function

import itertools
import unittest

from parameterized import parameterized

from openmdao.utils.assert_utils import assert_rel_error
import dymos.examples.min_time_climb_diff_inc.ex_min_time_climb_diff_inc as \
    ex_min_time_climb_diff_inc

try:
    from pyoptsparse import SNOPT
except ImportError:
    SNOPT = None


class TestExampleMinTimeClimbDiffInc(unittest.TestCase):

    @unittest.skipIf(SNOPT is None, 'This example requires pyoptsparse SNOPT')
    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['csc'],  # jacobian
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1]])
    )
    def test_results(self, transcription='gauss-lobatto', jacobian='csc'):
        ex_min_time_climb_diff_inc.SHOW_PLOTS = False
        p = ex_min_time_climb_diff_inc.min_time_climb_diff_inc(optimizer='SNOPT',
                                                               num_seg=5,
                                                               transcription_order=7,
                                                               transcription=transcription,
                                                               top_level_jacobian=jacobian)
        # Check that time matches to within 1% of an externally verified solution.
        assert_rel_error(self, p.model.phase0.get_values('time')[-1], 321.0, tolerance=0.01)

if __name__ == '__main__':
    unittest.main()
