import os
import unittest
from numpy.testing import assert_almost_equal

import dymos.examples.brachistochrone.test.ex_brachistochrone as ex_brachistochrone

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.testing_utils import use_tempdirs
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=True)


@use_tempdirs
class TestBrachistochroneExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def run_asserts(self, p):

        t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
        tf = p.get_val('traj0.phase0.timeseries.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.x')[0]
        xf = p.get_val('traj0.phase0.timeseries.x')[-1]

        y0 = p.get_val('traj0.phase0.timeseries.y')[0]
        yf = p.get_val('traj0.phase0.timeseries.y')[-1]

        v0 = p.get_val('traj0.phase0.timeseries.v')[0]
        vf = p.get_val('traj0.phase0.timeseries.v')[-1]

        g = p.get_val('traj0.phase0.parameter_vals:g')[0]

        thetaf = p.get_val('traj0.phase0.timeseries.theta')[-1]

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

        outputs = [op[1]['prom_name'] for op in p.model.list_outputs(out_stream=None, prom_name=True)]
        self.assertNotIn('traj0.phase0.timeseries.theta_rate', outputs)

    def test_ex_brachistochrone_radau_compressed(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        compressed=True)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_radau_uncompressed(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        compressed=False)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_gl_compressed(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        compressed=True)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_gl_uncompressed(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        compressed=False)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_shooting_gl_compressed(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='shooting-gauss-lobatto',
                                                        compressed=True)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_birkhoff(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='birkhoff',
                                                        num_segments=1, transcription_order=12)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_shooting_gl_uncompressed(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='shooting-gauss-lobatto',
                                                        compressed=False)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_shooting_radau_compressed(self):
        ex_brachistochrone.SHOW_PLOTS = True
        p = ex_brachistochrone.brachistochrone_min_time(transcription='shooting-radau',
                                                        optimizer='IPOPT',
                                                        force_alloc_complex=True,
                                                        compressed=True)
        self.run_asserts(p)
        self.tearDown()

    def test_ex_brachistochrone_shooting_radau_uncompressed(self):
        import dymos
        with dymos.options.temporary(include_check_partials=True):
            ex_brachistochrone.SHOW_PLOTS = True
            p = ex_brachistochrone.brachistochrone_min_time(transcription='shooting-radau',
                                                            optimizer='IPOPT',
                                                            compressed=False)
            self.run_asserts(p)
            self.tearDown()


if __name__ == '__main__':
    unittest.main()
