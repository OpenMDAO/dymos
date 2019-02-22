from __future__ import print_function, absolute_import, division

import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal

import dymos.examples.brachistochrone.ex_brachistochrone_vector_states as ex_brachistochrone_vs

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.assert_utils import assert_check_partials

OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


class TestBrachistochroneVectorStatesExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def assert_results(self, p):
        t_initial = p.get_val('phase0.time')[0]
        t_final = p.get_val('phase0.time')[-1]

        x0 = p.get_val('phase0.timeseries.states:pos')[0, 0]
        xf = p.get_val('phase0.timeseries.states:pos')[0, -1]

        y0 = p.get_val('phase0.timeseries.states:pos')[0, 1]
        yf = p.get_val('phase0.timeseries.states:pos')[-1, 1]

        v0 = p.get_val('phase0.timeseries.states:v')[0, 0]
        vf = p.get_val('phase0.timeseries.states:v')[-1, 0]

        g = p.get_val('phase0.timeseries.design_parameters:g')

        thetaf = p.get_val('phase0.timeseries.controls:theta')[-1, 0]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(t_final, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

    def assert_partials(self, p):
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_ex_brachistochrone_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           sim_record='ex_brachvs_radau_compressed.'
                                                                      'db',
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brach_radau_compressed.db'):
            os.remove('ex_brach_radau_compressed.db')

    def test_ex_brachistochrone_radau_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           sim_record='ex_brachvs_radau_'
                                                                      'uncompressed.db',
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brach_radau_uncompressed.db'):
            os.remove('ex_brach_radau_uncompressed.db')

    def test_ex_brachistochrone_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           sim_record='ex_brachvs_gl_compressed.db',
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brach_gl_compressed.db'):
            os.remove('ex_brach_gl_compressed.db')

    def test_ex_brachistochrone_gl_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           transcription_order=5,
                                                           compressed=False,
                                                           sim_record='ex_brachvs_gl_compressed.db',
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brach_gl_compressed.db'):
            os.remove('ex_brach_gl_compressed.db')


class TestBrachistochroneVectorStatesExampleSolveSegments(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def assert_results(self, p):
        t_initial = p.get_val('phase0.time')[0]
        t_final = p.get_val('phase0.time')[-1]

        x0 = p.get_val('phase0.timeseries.states:pos')[0, 0]
        xf = p.get_val('phase0.timeseries.states:pos')[0, -1]

        y0 = p.get_val('phase0.timeseries.states:pos')[0, 1]
        yf = p.get_val('phase0.timeseries.states:pos')[-1, 1]

        v0 = p.get_val('phase0.timeseries.states:v')[0, 0]
        vf = p.get_val('phase0.timeseries.states:v')[-1, 0]

        g = p.get_val('phase0.timeseries.design_parameters:g')

        thetaf = p.get_val('phase0.timeseries.controls:theta')[-1, 0]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(t_final, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

    def test_ex_brachistochrone_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           sim_record='ex_brachvs_radau_compressed.'
                                                                      'db',
                                                           force_alloc_complex=True,
                                                           solve_segments=True)

        p.final_setup()
        # set the final optimized control profile from
        # TestBrachistochroneVectorStatesExample.test_ex_brachistochrone_radau_compressed
        # and see if we get the right state history
        theta = np.array([2.54206362, 4.8278643, 10.11278149, 12.30024503, 17.35332815,
                          23.53948016, 25.30747573, 29.39010464, 35.47854735, 37.51549822,
                          42.16351471, 48.32419264, 50.21299389, 54.56658635, 60.77733663,
                          62.79222351, 67.35945157, 73.419141, 75.27851226, 79.60246558,
                          85.89170743, 87.96027845, 92.66164608, 98.89108826, ])

        p['phase0.controls:theta'] = theta.reshape((-1, 1))
        p['phase0.states:v'][:] = 100.  # bad initial guess on purpose
        p['phase0.states:v'][0] = 0.  # have to set the initial condition

        p['phase0.states:pos'][:] = 100.
        p['phase0.states:pos'][0, 0] = 0.  # have to set the initial condition
        p['phase0.states:pos'][0, 1] = 10.  # have to set the initial condition

        p['phase0.t_duration'] = 1.8016  # need the final duration (ivp style)

        p.run_model()
        self.assert_results(p)
        # self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brach_radau_compressed.db'):
            os.remove('ex_brach_radau_compressed.db')

    def test_ex_brachistochrone_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           sim_record='ex_brachvs_gl_compressed.db',
                                                           force_alloc_complex=True,
                                                           solve_segments=True)

        theta = np.array([1.04466973, 6.40253991, 12.26063396, 18.51810659, 25.07411252,
                          31.59842762, 37.76082779, 43.8810928, 50.27900244, 56.67270776,
                          62.78035981, 68.93138259, 75.45520008, 81.95935786, 88.05140149,
                          94.03879494, 100.22900215])

        p['phase0.controls:theta'] = theta.reshape((-1, 1))
        p['phase0.states:v'][:] = 100  # bad initial guess on purpose
        p['phase0.states:v'][0] = 0  # have to set the initial condition

        p['phase0.states:pos'][:] = 100
        p['phase0.states:pos'][0, 0] = 0  # have to set the initial condition
        p['phase0.states:pos'][0, 1] = 10.  # have to set the initial condition

        p['phase0.t_duration'] = 1.8016  # need the final duration (ivp style)

        p.run_model()
        self.assert_results(p)

        self.tearDown()
        if os.path.exists('ex_brach_gl_compressed.db'):
            os.remove('ex_brach_gl_compressed.db')

if __name__ == "__main__":
    unittest.main()
