
import unittest

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_rel_error


import dymos.examples.brachistochrone.ex_brachistochrone as ex_brachistochrone


class BenchmarkBrachistochrone(unittest.TestCase):
    """ Benchmarks for various permutations of the brachistochrone problem."""

    def run_asserts(self, p):
        t_initial = p.model.phase0.get_values('time')[0]
        tf = p.model.phase0.get_values('time')[-1]

        x0 = p.model.phase0.get_values('x')[0]
        xf = p.model.phase0.get_values('x')[-1]

        y0 = p.model.phase0.get_values('y')[0]
        yf = p.model.phase0.get_values('y')[-1]

        v0 = p.model.phase0.get_values('v')[0]
        vf = p.model.phase0.get_values('v')[-1]

        thetaf = p.model.phase0.get_values('theta')[-1]

        assert_rel_error(self, t_initial, 0.0)
        assert_rel_error(self, x0, 0.0)
        assert_rel_error(self, y0, 10.0)
        assert_rel_error(self, v0, 0.0)

        assert_rel_error(self, tf, 1.8016, tolerance=1.0E-3)
        assert_rel_error(self, xf, 10.0, tolerance=1.0E-3)
        assert_rel_error(self, yf, 5.0, tolerance=1.0E-3)
        assert_rel_error(self, vf, 9.902, tolerance=1.0E-3)

        assert_rel_error(self, thetaf, 100.12, tolerance=0.03)

    def benchmark_radau_30_3_color_simul_compressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=True,
                                                        dynamic_simul_derivs=True)
        self.run_asserts(p)

    def benchmark_radau_30_3_color_simul_uncompressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=False,
                                                        dynamic_simul_derivs=True)
        self.run_asserts(p)

    def benchmark_radau_30_3_nocolor_nosimul_compressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='radau-ps',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=True,
                                                        dynamic_simul_derivs=False,
                                                        color_file=None)
        self.run_asserts(p)

    def benchmark_gl_30_3_color_simul_compressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=True,
                                                        dynamic_simul_derivs=True)
        self.run_asserts(p)

    def benchmark_gl_30_3_color_simul_uncompressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=False,
                                                        dynamic_simul_derivs=True)
        self.run_asserts(p)

    def benchmark_gl_30_3_nocolor_nosimul_compressed_snopt(self):
        ex_brachistochrone.SHOW_PLOTS = False
        p = ex_brachistochrone.brachistochrone_min_time(transcription='gauss-lobatto',
                                                        optimizer='SNOPT',
                                                        num_segments=30,
                                                        transcription_order=3,
                                                        compressed=True,
                                                        dynamic_simul_derivs=False,
                                                        color_file=None)
        self.run_asserts(p)
