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
                                                           force_alloc_complex=True)
        p.run_driver()

        print('v', p['phase0.states:v'])
        print('pos', p['phase0.states:pos'])
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
                                                           force_alloc_complex=True)
        p.run_driver()
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
                                                           force_alloc_complex=True)

        p.run_driver()
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
                                                           force_alloc_complex=True)
        p.run_driver()
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

        v = np.array([0., 0.7826936, 1.85519827, 2.19133903, 2.94888909, 3.96188083, 4.27352452, 4.97004106,
                      5.88315586, 6.15847324, 6.7602533, 7.52142417, 7.74480064, 8.22334063, 8.80196882, 8.9638606,
                      9.29404116, 9.65664974, 9.74897797, 9.91965675, 10.05680102, 10.07516101, 10.07070724,
                      9.9614451, 9.9028538])

        pos = np.array([[0.00000000e+00, 1.00000000e+01],
                        [1.94778856e-03, 9.96869875e+00],
                        [2.17495404e-02, 9.82460935e+00],
                        [3.52157241e-02, 9.75516795e+00],
                        [8.85025140e-02, 9.55658605e+00],
                        [2.22886301e-01, 9.19975901e+00],
                        [2.82537968e-01, 9.06884572e+00],
                        [4.52556485e-01, 8.74051469e+00],
                        [7.74071639e-01, 8.23539137e+00],
                        [8.99200543e-01, 8.06627214e+00],
                        [1.23058548e+00, 7.66989707e+00],
                        [1.79088189e+00, 7.11563589e+00],
                        [1.99299275e+00, 6.94177276e+00],
                        [2.49901192e+00, 6.55216909e+00],
                        [3.29293410e+00, 6.04988562e+00],
                        [3.56638899e+00, 5.90325005e+00],
                        [4.22869168e+00, 5.59593864e+00],
                        [5.21411520e+00, 5.24545843e+00],
                        [5.54041544e+00, 5.15417801e+00],
                        [6.30817770e+00, 4.98306345e+00],
                        [7.40164406e+00, 4.84326352e+00],
                        [7.75273244e+00, 4.82448866e+00],
                        [8.55830403e+00, 4.82912543e+00],
                        [9.65834602e+00, 4.94058042e+00],
                        [1.00000000e+01, 5.00000000e+00]])



        p['phase0.controls:theta'] = theta.reshape(-1, 1)
        # p['phase0.states:v'] = v.reshape(-1, 1)
        p['phase0.states:v'][:] = 100
        p['phase0.states:v'][0] = 0

        p['phase0.states:pos'][:] = 100
        p['phase0.states:pos'][0,0] = 0
        p['phase0.states:pos'][0,1] = 10.

        print('foo', p['phase0.states:pos'])
        # p['phase0.states:pos'] = pos
        p['phase0.t_duration'] = 1.8016

        p.run_model()
        print('bar', p['phase0.states:pos'])
        # p.final_setup()
        # p.model.run_apply_nonlinear()
        # p.model.phase0.collocation_constraint.list_outputs(residuals=True, print_arrays=True)
        self.assert_results(p)
        # self.assert_partials(p)
        self.tearDown()
        if os.path.exists('ex_brach_radau_compressed.db'):
            os.remove('ex_brach_radau_compressed.db')

    # def test_ex_brachistochrone_radau_uncompressed(self):
    #     ex_brachistochrone_vs.SHOW_PLOTS = True
    #     p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
    #                                                        compressed=False,
    #                                                        sim_record='ex_brachvs_radau_'
    #                                                                   'uncompressed.db',
    #                                                        force_alloc_complex=True)
    #     self.assert_results(p)
    #     self.assert_partials(p)
    #     self.tearDown()
    #     if os.path.exists('ex_brach_radau_uncompressed.db'):
    #         os.remove('ex_brach_radau_uncompressed.db')

    # def test_ex_brachistochrone_gl_compressed(self):
    #     ex_brachistochrone_vs.SHOW_PLOTS = True
    #     p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
    #                                                        compressed=True,
    #                                                        sim_record='ex_brachvs_gl_compressed.db',
    #                                                        force_alloc_complex=True)
    #     self.assert_results(p)
    #     self.assert_partials(p)
    #     self.tearDown()
    #     if os.path.exists('ex_brach_gl_compressed.db'):
    #         os.remove('ex_brach_gl_compressed.db')

    # def test_ex_brachistochrone_gl_uncompressed(self):
    #     ex_brachistochrone_vs.SHOW_PLOTS = True
    #     p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
    #                                                        transcription_order=5,
    #                                                        compressed=False,
    #                                                        sim_record='ex_brachvs_gl_compressed.db',
    #                                                        force_alloc_complex=True)
    #     self.assert_results(p)
    #     self.assert_partials(p)
    #     self.tearDown()
    #     if os.path.exists('ex_brach_gl_compressed.db'):
    #         os.remove('ex_brach_gl_compressed.db')


if __name__ == "__main__":
    unittest.main()
