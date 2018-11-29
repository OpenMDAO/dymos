from __future__ import print_function, absolute_import, division

import os
import unittest
from numpy.testing import assert_almost_equal

from parameterized import parameterized
from itertools import product

import dymos.examples.brachistochrone.ex_brachistochrone_vector_states as ex_brachistochrone_vs

from openmdao.utils.general_utils import set_pyoptsparse_opt
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


class TestBrachistochroneVectorStatesExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def run_asserts(self, p):
        t_initial = p.model.phase0.get_values('time')[0]
        tf = p.model.phase0.get_values('time')[-1]

        x0 = p.model.phase0.get_values('pos')[0, 0]
        xf = p.model.phase0.get_values('pos')[0, -1]

        y0 = p.model.phase0.get_values('pos')[-1, 0]
        yf = p.model.phase0.get_values('pos')[-1, -1]

        v0 = p.model.phase0.get_values('v')[0]
        vf = p.model.phase0.get_values('v')[-1]

        g = p.model.phase0.get_values('g')

        thetaf = p.model.phase0.get_values('theta')[-1]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(tf, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

    def test_ex_brachistochrone_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           sim_record='ex_brachvs_radau_compressed.'
                                                                      'db')
        self.run_asserts(p)
        self.tearDown()
        if os.path.exists('ex_brach_radau_compressed.db'):
            os.remove('ex_brach_radau_compressed.db')

    def test_ex_brachistochrone_radau_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           sim_record='ex_brachvs_radau_'
                                                                      'uncompressed.db')
        self.run_asserts(p)
        self.tearDown()
        if os.path.exists('ex_brach_radau_uncompressed.db'):
            os.remove('ex_brach_radau_uncompressed.db')

    def test_ex_brachistochrone_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           sim_record='ex_brachvs_gl_compressed.db')
        self.run_asserts(p)
        self.tearDown()
        if os.path.exists('ex_brach_gl_compressed.db'):
            os.remove('ex_brach_gl_compressed.db')

    def test_ex_brachistochrone_gl_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=False,
                                                           sim_record='ex_brachvs_gl_compressed.db')
        self.run_asserts(p)
        self.tearDown()
        if os.path.exists('ex_brach_gl_compressed.db'):
            os.remove('ex_brach_gl_compressed.db')
