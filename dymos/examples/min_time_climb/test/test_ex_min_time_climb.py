
from __future__ import absolute_import, division, print_function

import os
import itertools
import unittest
from shutil import rmtree
from tempfile import mkdtemp
import errno

import numpy as np
from numpy.testing import assert_almost_equal

from parameterized import parameterized

from openmdao.utils.assert_utils import assert_rel_error
import dymos.examples.min_time_climb.ex_min_time_climb as ex_min_time_climb

SHOW_PLOTS = False


class TestExampleMinTimeClimb(unittest.TestCase):

    def setUp(self):
        self.orig_dir = os.getcwd()
        self.temp_dir = mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.orig_dir)
        try:
            rmtree(self.temp_dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0]])
    )
    def test_results(self, transcription='gauss-lobatto'):
        p = ex_min_time_climb.min_time_climb(optimizer='SLSQP',
                                             num_seg=12,
                                             transcription_order=3,
                                             transcription=transcription)

        phase = p.model.phase0

        # Check that time matches to within 1% of an externally verified solution.
        assert_rel_error(self, phase.get_values('time')[-1], 321.0, tolerance=2)

if __name__ == '__main__':
    unittest.main()
