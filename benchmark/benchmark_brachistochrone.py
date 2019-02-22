
import unittest

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_rel_error


import dymos.examples.brachistochrone.ex_brachistochrone as ex_brachistochrone


class BenchmarkBrachistochrone(unittest.TestCase):
    """ Benchmarks for various permutations of the brachistochrone problem."""

    def run_asserts(self, p):

        t_initial = p.get_val('phase0.timeseries.time')[0]
        tf = p.get_val('phase0.timeseries.time')[-1]

        x0 = p.get_val('phase0.timeseries.states:x')[0]
        xf = p.get_val('phase0.timeseries.states:x')[-1]

        y0 = p.get_val('phase0.timeseries.states:y')[0]
        yf = p.get_val('phase0.timeseries.states:y')[-1]

        v0 = p.get_val('phase0.timeseries.states:v')[0]
        vf = p.get_val('phase0.timeseries.states:v')[-1]

        g = p.get_val('phase0.timeseries.input_parameters:g')[0]

        thetaf = p.get_val('phase0.timeseries.controls:theta')[-1]

        assert_rel_error(self, t_initial, 0.0)
        assert_rel_error(self, x0, 0.0)
        assert_rel_error(self, y0, 10.0)
        assert_rel_error(self, v0, 0.0)

        assert_rel_error(self, tf, 1.8016, tolerance=0.0001)
        assert_rel_error(self, xf, 10.0, tolerance=0.001)
        assert_rel_error(self, yf, 5.0, tolerance=0.001)
        assert_rel_error(self, vf, 9.902, tolerance=0.001)
        assert_rel_error(self, g, 9.80665, tolerance=0.001)

        assert_rel_error(self, thetaf, 100.12, tolerance=1)

    def benchmark_radau_30_3_color_simul_compressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=True)
        self.run_asserts(p)

    def benchmark_radau_30_3_color_simul_uncompressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=False)
        self.run_asserts(p)

    def benchmark_gl_30_3_color_simul_compressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=True)
        self.run_asserts(p)

    def benchmark_gl_30_3_color_simul_uncompressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=False)
        self.run_asserts(p)
