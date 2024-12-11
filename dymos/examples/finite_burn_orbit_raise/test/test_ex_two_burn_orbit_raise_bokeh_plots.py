import importlib
import os
import shutil
import unittest
import pathlib

try:
    import matplotlib
except ImportError:
    matplotlib = None

from openmdao.utils.testing_utils import require_pyoptsparse
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars_context

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem
from dymos.utils.testing_utils import _get_reports_dir

_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)
bokeh_available = importlib.util.find_spec('bokeh') is not None


@use_tempdirs
@require_pyoptsparse(optimizer='SLSQP')
class TestExampleTwoBurnOrbitRaise(unittest.TestCase):

    def setUp(self):
        # We need to remove the TESTFLO_RUNNING environment variable for reports to be generated.
        # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything if set
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)

    def tearDown(self):
        if os.path.isdir('plots'):
            shutil.rmtree('plots')

        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    @unittest.skipIf(not bokeh_available, 'bokeh unavailable')
    def test_bokeh_plots(self):
        with set_env_vars_context(OPENMDAO_REPORTS='1'):
            with dm.options.temporary(plots='bokeh'):
                p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                                 compressed=False, optimizer='SLSQP', show_output=False,
                                                 run_driver=False)

                html_file = pathlib.Path(_get_reports_dir(p)) / 'traj_results_report.html'
                self.assertTrue(html_file.exists(), msg=f'{html_file} does not exist!')

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_mpl_plots(self):
        with dm.options.temporary(plots='matplotlib'):
            p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                             compressed=False, optimizer='SLSQP', show_output=False,
                                             run_driver=False)

            expected_files = ('deltav.png', 'r.png', 'accel.png',
                              'u1.png', 'vr.png', 'pos_x.png',
                              'vt.png', 'pos_y.png', 'theta.png')

            for file in expected_files:
                plotfile = pathlib.Path(p.get_reports_dir(p)).joinpath('plots') / file
                self.assertTrue(plotfile.exists(), msg=f'{plotfile} does not exist!')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
