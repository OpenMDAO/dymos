import importlib
import os
import pathlib
import unittest
from numpy.testing import assert_almost_equal

import sys

from openmdao.utils.general_utils import set_pyoptsparse_opt, printoptions
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars_context
from openmdao.utils.tests.test_hooks import hooks_active

import dymos as dm
import dymos.examples.brachistochrone.test.ex_brachistochrone_vector_states as ex_brachistochrone_vs
from dymos.utils.testing_utils import assert_check_partials, _get_reports_dir

bokeh_available = importlib.util.find_spec('bokeh') is not None

OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


@use_tempdirs
class TestBrachistochroneVectorStatesExample(unittest.TestCase):

    def setUp(self):
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    def assert_results(self, p):
        t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
        t_final = p.get_val('traj0.phase0.timeseries.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.pos')[0, 0]
        xf = p.get_val('traj0.phase0.timeseries.pos')[0, -1]

        y0 = p.get_val('traj0.phase0.timeseries.pos')[0, 1]
        yf = p.get_val('traj0.phase0.timeseries.pos')[-1, 1]

        v0 = p.get_val('traj0.phase0.timeseries.v')[0, 0]
        vf = p.get_val('traj0.phase0.timeseries.v')[-1, 0]

        g = p.get_val('traj0.phase0.parameter_vals:g')

        thetaf = p.get_val('traj0.phase0.timeseries.theta')[-1, 0]

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
            cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)
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

    def test_ex_brachistochrone_vs_birkhoff(self):
        ex_brachistochrone_vs.SHOW_PLOTS = True
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='birkhoff',
                                                           transcription_order=12,
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           run_driver=True)

        self.assert_results(p)
        self.assert_partials(p)

    @unittest.skipIf(not bokeh_available, 'bokeh unavailable')
    @hooks_active
    def test_bokeh_plots(self):
        with set_env_vars_context(OPENMDAO_REPORTS='1'):
            with dm.options.temporary(plots='bokeh'):
                p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                                   compressed=False,
                                                                   force_alloc_complex=True,
                                                                   run_driver=True,
                                                                   simulate=True,
                                                                   make_plots=True)

                self.assert_results(p)
                self.assert_partials(p)

                html_file = pathlib.Path(_get_reports_dir(p)) / 'traj0_results_report.html'
                self.assertTrue(html_file.exists(), msg=f'{html_file} does not exist!')

                with open(html_file) as f:
                    html_data = f.read()

                expected_labels = ['"axis_label":"pos[0] (m)"',
                                   '"axis_label":"pos[1] (m)"',
                                   '"axis_label":"v (m/s)"',
                                   '"axis_label":"theta (deg)"']

                for label in expected_labels:
                    self.assertIn(label, html_data)

                self.assertNotIn('"axis_label":"pos (m)"', html_data)


if __name__ == "__main__":
    unittest.main()
