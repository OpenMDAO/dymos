import unittest
from numpy.testing import assert_almost_equal

from openmdao.utils.general_utils import set_pyoptsparse_opt, printoptions
from openmdao.utils.testing_utils import use_tempdirs

import dymos.examples.brachistochrone.test.ex_brachistochrone_vector_states as ex_brachistochrone_vs
from dymos.utils.testing_utils import assert_check_partials

OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


@use_tempdirs
class TestBrachistochroneVectorStatesExample(unittest.TestCase):

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

    def assert_partials(self, p):
        with printoptions(linewidth=1024, edgeitems=100):
            cpd = p.check_partials(method='cs', compact_print=True)
        assert_check_partials(cpd)

    def test_ex_brachistochrone_vs_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           run_driver=False)
        p.run_driver()
        self.assert_partials(p)

    def test_ex_brachistochrone_vs_radau_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)

    def test_ex_brachistochrone_vs_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           run_driver=True)

        self.assert_results(p)
        self.assert_partials(p)

    def test_ex_brachistochrone_vs_gl_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           transcription_order=5,
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           run_driver=True)
        self.assert_results(p)
        self.assert_partials(p)


if __name__ == "__main__":
    unittest.main()
