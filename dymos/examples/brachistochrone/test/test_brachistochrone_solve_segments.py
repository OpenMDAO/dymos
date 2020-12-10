import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import openmdao.api as om
import dymos as dm

import dymos.examples.brachistochrone.test.ex_brachistochrone_vector_states as ex_brachistochrone_vs
import dymos.examples.brachistochrone.test.ex_brachistochrone as ex_brachistochrone
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.utils.general_utils import set_pyoptsparse_opt

OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')


def _make_problem(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                  compressed=True, optimizer='SLSQP', run_driver=True, force_alloc_complex=False,
                  solve_segments=False):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if optimizer == 'SNOPT':
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['print_level'] = 5
    p.driver.declare_coloring(tol=1.0E-12)

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
    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    p.model.add_subsystem('traj0', traj)
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                    fix_initial=False, fix_final=False, solve_segments=solve_segments)
    phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                    fix_initial=False, fix_final=False, solve_segments=solve_segments)

    # Note that by omitting the targets here Dymos will automatically attempt to connect
    # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
    phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                    fix_initial=False, fix_final=False, solve_segments=solve_segments)

    phase.add_control('theta',
                      continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', targets=['g'], units='m/s**2')

    phase.add_boundary_constraint('x', loc='initial', equals=0)
    phase.add_boundary_constraint('y', loc='initial', equals=10)
    phase.add_boundary_constraint('v', loc='initial', equals=0)

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 2.0

    p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
    p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
    p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
    p['traj0.phase0.parameters:g'] = 9.80665

    # dm.run_problem(p, run_driver=run_driver, simulate=True, make_plots=False)

    return p


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
class TestBrachistochroneSolveSegments(unittest.TestCase):

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
        assert_almost_equal(x0, 0.0, decimal=4)
        assert_almost_equal(y0, 10.0, decimal=4)
        assert_almost_equal(v0, 0.0, decimal=4)

        assert_almost_equal(t_final, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

    def test_brachistochrone_solve_segments(self):

        for tx in ('radau-ps', 'gauss-lobatto'):
            for solve_segs in (True, False, 'forward', 'backward', None):
                for compressed in (True, False):
                    print(f'transcription: {tx}  solve_segments: {solve_segs}  compressed: {compressed}')
                    with self.subTest(f'transcription: {tx}  solve_segments: {solve_segs}  '
                                      f'compressed: {compressed}'):
                        p = _make_problem(transcription=tx,
                                          compressed=compressed,
                                          optimizer='SLSQP',
                                          force_alloc_complex=True,
                                          solve_segments=solve_segs,
                                          num_segments=20,
                                          transcription_order=3)
                        dm.run_problem(p)
                        self.assert_results(p)

    def _print_states(self, solve_segments, output_dict, state_idx_map):
        for state in ['x', 'y', 'v']:
            print(state)
            vals = output_dict[f'states:{state}']['value']
            resids = output_dict[f'states:{state}']['resids']
            defects = output_dict[f'collocation_constraint.defects:{state}']['value']
            out = np.zeros((len(vals), 4))
            out[:, 0] = vals[:, 0]
            out[:, 1] = resids[:, 0]
            # if solve_segments in {True, 'forward'}:
            #     out[1:, 2] = defects[:, 0]
            # elif solve_segments == 'backward':
            #     out[:-1, 2] = defects[:, 0]
            out[state_idx_map[state]['solver'], 2] = defects[:, 0]
            out[:, 3] = out[:, 2] - out[:, 1]
            print(out)

    def test_brachistochrone_solve_segments2(self):

        ss = 'backward'

        p = _make_problem(transcription='radau-ps',
                          compressed=False,
                          optimizer='SLSQP',
                          force_alloc_complex=True,
                          solve_segments=ss,
                          num_segments=5,
                          transcription_order=3,)
        p.model.traj0.phases.phase0.nonlinear_solver.options['maxiter'] = 0
        state_idx_map = p.model.traj0.phases.phase0.options['transcription'].state_idx_map
        p.run_model()
        # p.model.traj0.phases.phase0.collocation_constraint.list_outputs(print_arrays=True)
        # p.model.traj0.phases.phase0.indep_states.list_outputs(residuals=True, print_arrays=True)
        outputs = p.model.traj0.phases.phase0.list_outputs(residuals=True, print_arrays=True, prom_name=True, out_stream=None)
        output_dict = {d['prom_name']: {'value': d['value'], 'resids': d['resids']} for path, d in outputs}
        self._print_states(ss, output_dict, state_idx_map)

        p.model.traj0.phases.phase0.nonlinear_solver.options['maxiter'] = 100
        p.run_driver()
        self.assert_results(p)

        outputs = p.model.traj0.phases.phase0.list_outputs(residuals=True, print_arrays=True, prom_name=True, out_stream=None)
        output_dict = {d['prom_name']: {'value': d['value'], 'resids': d['resids']} for path, d in outputs}
        self._print_states(ss, output_dict, state_idx_map)
