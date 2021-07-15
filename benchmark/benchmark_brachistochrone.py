
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.testing_utils import use_tempdirs
import dymos as dm
from dymos.examples.brachistochrone import BrachistochroneODE


def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=True, optimizer='SLSQP', simul_derivs=True):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    OPT, OPTIMIZER = set_pyoptsparse_opt(optimizer, fallback=True)
    p.driver.options['optimizer'] = OPTIMIZER

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

    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

    phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='xdot')
    phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m', rate_source='ydot')
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False,
                    units='m/s', rate_source='vdot')

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', units='m/s**2', val=9.80665, targets=['g'])

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    p['phase0.states:x'] = phase.interp('x', ys=[0, 10])
    p['phase0.states:y'] = phase.interp('y', ys=[10, 5])
    p['phase0.states:v'] = phase.interp('v', ys=[0, 9.9])
    p['phase0.controls:theta'] = phase.interp('theta', ys=[5, 100])
    p['phase0.parameters:g'] = 9.80665

    p.run_driver()

    return p


@use_tempdirs
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

        g = p.get_val('phase0.timeseries.parameters:g')[0]

        thetaf = p.get_val('phase0.timeseries.controls:theta')[-1]

        assert_near_equal(t_initial, 0.0)
        assert_near_equal(x0, 0.0)
        assert_near_equal(y0, 10.0)
        assert_near_equal(v0, 0.0)

        assert_near_equal(tf, 1.8016, tolerance=0.0001)
        assert_near_equal(xf, 10.0, tolerance=0.001)
        assert_near_equal(yf, 5.0, tolerance=0.001)
        assert_near_equal(vf, 9.902, tolerance=0.001)
        assert_near_equal(g, 9.80665, tolerance=0.001)

        assert_near_equal(thetaf, 100.12, tolerance=1)

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
