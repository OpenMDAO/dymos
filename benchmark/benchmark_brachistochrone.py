
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
import dymos as dm
from dymos.examples.brachistochrone import BrachistochroneODE


def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=True, optimizer='SLSQP', simul_derivs=True):
    p = om.Problem(model=om.Group())

    # if optimizer == 'SNOPT':
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer

    if simul_derivs:
        p.driver.declare_coloring()

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)
    elif transcription == 'runge-kutta':
        t = dm.RungeKutta(num_segments=num_segments,
                          order=transcription_order,
                          compressed=compressed)

    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=False)
    phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=False)
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False)

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_input_parameter('g', units='m/s**2', val=9.80665)

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
    p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
    p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
    p['phase0.input_parameters:g'] = 9.80665

    p.run_driver()

    return p


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
        p = brachistochrone_min_time(transcription='radau-ps',
                                     optimizer='SNOPT',
                                     num_segments=30,
                                     transcription_order=3,
                                     compressed=True)
        self.run_asserts(p)

    def benchmark_radau_30_3_color_simul_uncompressed_snopt(self):
        p = brachistochrone_min_time(transcription='radau-ps',
                                     optimizer='SNOPT',
                                     num_segments=30,
                                     transcription_order=3,
                                     compressed=False)
        self.run_asserts(p)

    def benchmark_gl_30_3_color_simul_compressed_snopt(self):
        p = brachistochrone_min_time(transcription='gauss-lobatto',
                                     optimizer='SNOPT',
                                     num_segments=30,
                                     transcription_order=3,
                                     compressed=True)
        self.run_asserts(p)

    def benchmark_gl_30_3_color_simul_uncompressed_snopt(self):
        p = brachistochrone_min_time(transcription='gauss-lobatto',
                                     optimizer='SNOPT',
                                     num_segments=30,
                                     transcription_order=3,
                                     compressed=False)
        self.run_asserts(p)

    def benchmark_gl_30_3_color_simul_compressed_rk4(self):
        p = brachistochrone_min_time(transcription='runge-kutta',
                                     optimizer='SNOPT',
                                     num_segments=30,
                                     transcription_order=3,
                                     compressed=True)
        self.run_asserts(p)
