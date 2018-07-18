from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.examples.finite_burn_orbit_raise.ex_two_burn_orbit_raise import \
    two_burn_orbit_raise_problem


class TestExampleTwoBurnOrbitRaise(unittest.TestCase):

    def test_ex_two_burn_orbit_raise(self):

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=True)

        assert_rel_error(self, p.get_val('burn2.states:deltav')[-1], 0.3995, tolerance=1.0E-3)

    def test_partials(self):

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=True)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)
