import unittest
from numpy.testing import assert_almost_equal

import dymos.examples.brachistochrone.test.ex_brachistochrone_vector_states as ex_brachistochrone_vs
import dymos.examples.brachistochrone.test.ex_brachistochrone as ex_brachistochrone
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.utils.general_utils import set_pyoptsparse_opt

OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')


@use_tempdirs
class TestBrachistochroneVectorStatesExampleSolveSegments(unittest.TestCase):

    def assert_results(self, p):
        t_initial = p.get_val('traj0.phase0.time')[0]
        t_final = p.get_val('traj0.phase0.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.states:pos')[0, 0]
        xf = p.get_val('traj0.phase0.timeseries.states:pos')[0, -1]

        y0 = p.get_val('traj0.phase0.timeseries.states:pos')[0, 1]
        yf = p.get_val('traj0.phase0.timeseries.states:pos')[-1, 1]

        v0 = p.get_val('traj0.phase0.timeseries.states:v')[0, 0]
        vf = p.get_val('traj0.phase0.timeseries.states:v')[-1, 0]

        g = p.get_val('traj0.phase0.timeseries.parameters:g')

        thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1, 0]

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

    def test_ex_brachistochrone_vs_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments=True,
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)


@use_tempdirs
class TestBrachistochroneExampleSolveSegments(unittest.TestCase):

    def assert_results(self, p):
        t_initial = p.get_val('traj0.phase0.time')[0]
        t_final = p.get_val('traj0.phase0.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.states:x')[0]
        xf = p.get_val('traj0.phase0.timeseries.states:x')[-1]

        y0 = p.get_val('traj0.phase0.timeseries.states:y')[0]
        yf = p.get_val('traj0.phase0.timeseries.states:y')[-1]

        v0 = p.get_val('traj0.phase0.timeseries.states:v')[0]
        vf = p.get_val('traj0.phase0.timeseries.states:v')[-1]

        g = p.get_val('traj0.phase0.timeseries.parameters:g')

        thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1, 0]

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

    def test_ex_brachistochrone_vs_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        compressed=True,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=10,
                                                        transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        compressed=True,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=10,
                                                        transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        compressed=False,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=10,
                                                        transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        compressed=False,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=10,
                                                        transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        compressed=True,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=1,
                                                        transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        compressed=True,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=1,
                                                        transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        compressed=False,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=1,
                                                        transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        compressed=False,
                                                        force_alloc_complex=True,
                                                        solve_segments=True,
                                                        num_segments=1,
                                                        transcription_order=11)
        self.assert_results(p)
