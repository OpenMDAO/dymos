from __future__ import print_function, absolute_import, division

import unittest
from numpy.testing import assert_almost_equal

from parameterized import parameterized

import dymos.examples.double_integrator.ex_double_integrator as ex_double_integrator


class TestDoubleIntegratorExample(unittest.TestCase):

    @parameterized.expand(['gauss-lobatto', 'radau-ps'])
    def test_ex_double_integrator(self, transcription='radau-ps'):
        ex_double_integrator.SHOW_PLOTS = False
        p = ex_double_integrator.double_integrator_direct_collocation(transcription)

        x0 = p.model.phase0.get_values('x')[0]
        xf = p.model.phase0.get_values('x')[-1]

        v0 = p.model.phase0.get_values('v')[0]
        vf = p.model.phase0.get_values('v')[-1]

        assert_almost_equal(x0, 0.0)
        assert_almost_equal(xf, 0.25)

        assert_almost_equal(v0, 0.0)
        assert_almost_equal(vf, 0.0)

