from __future__ import print_function, absolute_import, division

import unittest
from numpy.testing import assert_almost_equal

from parameterized import parameterized

import dymos.examples.brachistochrone.ex_brachistochrone as ex_brachistochrone


class TestBrachistochroneExample(unittest.TestCase):

    # @parameterized.expand(['gauss-lobatto', 'radau-ps'])
    def test_ex_brachistochrone(self, transcription='radau-ps'):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription)

        t_initial = p.model.phase0.get_values('time')[0]
        tf = p.model.phase0.get_values('time')[-1]

        x0 = p.model.phase0.get_values('x')[0]
        xf = p.model.phase0.get_values('x')[-1]

        y0 = p.model.phase0.get_values('y')[0]
        yf = p.model.phase0.get_values('y')[-1]

        v0 = p.model.phase0.get_values('v')[0]
        vf = p.model.phase0.get_values('v')[-1]

        thetaf = p.model.phase0.get_values('theta')[-1]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(tf, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)

        if transcription != 'radau-ps':
            assert_almost_equal(thetaf, 100.12, decimal=1)
