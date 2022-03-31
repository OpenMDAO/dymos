import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm

import dymos.examples.brachistochrone.test.ex_brachistochrone_vector_states as ex_brachistochrone_vs
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from openmdao.utils.general_utils import set_pyoptsparse_opt

OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')


def _make_problem(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                  compressed=True, optimizer='SLSQP', force_alloc_complex=False,
                  solve_segments=False, y_bounds=None):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if optimizer == 'SNOPT':
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['mu_init'] = 1e-3
        p.driver.opt_settings['max_iter'] = 500
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['mu_strategy'] = 'monotone'
    p.driver.declare_coloring(tol=1.0E-12)

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    p.model.add_subsystem('traj0', traj)
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.add_state('x', fix_initial=False, fix_final=False, solve_segments=solve_segments)
    phase.add_state('y', fix_initial=False, fix_final=False, solve_segments=solve_segments)

    if y_bounds is not None:
        phase.set_state_options('y', lower=y_bounds[0], upper=y_bounds[1])

    # Note that by omitting the targets here Dymos will automatically attempt to connect
    # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
    phase.add_state('v', fix_initial=False, fix_final=False, solve_segments=solve_segments)

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

    p.set_solver_print(0)

    p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 1.5

    p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
    p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
    p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
    p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
    p['traj0.phase0.parameters:g'] = 9.80665

    return p


@require_pyoptsparse(optimizer='SLSQP')
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

        g = p.get_val('traj0.phase0.parameter_vals:g')

        thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1, 0]

        assert_near_equal(t_initial, 0.0, tolerance=1.0E-2)
        assert_near_equal(x0, 0.0, tolerance=1.0E-2)
        assert_near_equal(y0, 10.0, tolerance=1.0E-2)
        assert_near_equal(v0, 0.0, tolerance=1.0E-2)

        assert_near_equal(t_final, 1.8016, tolerance=1.0E-2)
        assert_near_equal(xf, 10.0, tolerance=1.0E-2)
        assert_near_equal(yf, 5.0, tolerance=1.0E-2)
        assert_near_equal(vf, 9.902, tolerance=1.0E-2)
        assert_near_equal(g, 9.80665, tolerance=1.0E-2)

        assert_near_equal(thetaf, 100.12, tolerance=1.0E-2)

    def test_ex_brachistochrone_vs_radau_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_compressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_uncompressed(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=10,
                                                           transcription_order=3)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=True,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_radau_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='radau-ps',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)

    def test_ex_brachistochrone_vs_gl_single_segment(self):
        ex_brachistochrone_vs.SHOW_PLOTS = False
        p = ex_brachistochrone_vs.brachistochrone_min_time(transcription='gauss-lobatto',
                                                           compressed=False,
                                                           force_alloc_complex=True,
                                                           solve_segments='forward',
                                                           num_segments=1,
                                                           transcription_order=11)
        self.assert_results(p)


@require_pyoptsparse(optimizer='IPOPT')
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

        g = p.get_val('traj0.phase0.parameter_vals:g')

        thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1, 0]

        assert_near_equal(t_initial, 0.0, tolerance=1.0E-2)
        assert_near_equal(x0, 0.0, tolerance=1.0E-2)
        assert_near_equal(y0, 10.0, tolerance=1.0E-2)
        assert_near_equal(v0, 0.0, tolerance=1.0E-2)

        assert_near_equal(t_final, 1.8016, tolerance=1.0E-2)
        assert_near_equal(xf, 10.0, tolerance=1.0E-2)
        assert_near_equal(yf, 5.0, tolerance=1.0E-2)
        assert_near_equal(vf, 9.902, tolerance=1.0E-2)
        assert_near_equal(g, 9.80665, tolerance=1.0E-2)

        assert_near_equal(thetaf, 100.12, tolerance=1.0E-2)

    def test_brachistochrone_solve_segments_radau_False_compressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=False,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_radau_forward_compressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='forward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_radau_backward_compressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='backward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_radau_None_compressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=None,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_radau_False_noncompressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=False,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_radau_forward_noncompressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='forward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_radau_backward_noncompressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='backward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_radau_None_noncompressed(self):
        p = _make_problem(transcription='radau-ps',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=None,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_False_compressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=False,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_forward_compressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='forward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_backward_compressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='backward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_None_compressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=None,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_False_noncompressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=False,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_forward_noncompressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='forward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_backward_noncompressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='backward',
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_solve_segments_gl_None_noncompressed(self):
        p = _make_problem(transcription='gauss-lobatto',
                          compressed=False,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments=None,
                          num_segments=8,
                          transcription_order=3)
        dm.run_problem(p)
        self.assert_results(p)

    def test_brachistochrone_bounded_solve_segments(self):

        expected_warning = '<class \'openmdao.utils.om_warnings.UnusedOptionWarning\'>: State y ' \
                           'has bounds but they are not enforced when using ' \
                           '`solve_segments.` Apply a path constraint to y to enforce bounds.'

        # Setup the problem
        with warnings.catch_warnings(record=True) as ctx:
            _make_problem(transcription='radau-ps',
                          compressed=True,
                          optimizer='IPOPT',
                          force_alloc_complex=True,
                          solve_segments='forward',
                          num_segments=8,
                          transcription_order=3,
                          y_bounds=(5, 10))

            warnings.simplefilter('always')
            self.assertIn(expected_warning, [str(w.message) for w in ctx])
