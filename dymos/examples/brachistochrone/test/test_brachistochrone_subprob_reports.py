"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import os
from packaging.version import Version

import openmdao.api as om
import openmdao.core.problem
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.tests.test_hooks import hooks_active
from openmdao.visualization.n2_viewer.n2_viewer import _default_n2_filename
from openmdao.visualization.scaling_viewer.scaling_report import _default_scaling_filename
from openmdao import __version__ as openmdao_version


import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


def setup_model_radau(do_reports):
    p = om.Problem(model=om.Group())

    p.driver = om.ScipyOptimizeDriver()
    p.driver.declare_coloring(tol=1.0E-12)

    t = dm.Radau(num_segments=10, order=3)

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    p.model.add_subsystem('traj0', traj)
    traj.add_phase('phase0', phase)

    # p.model.add_subsystem('traj0', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.add_state('x', fix_initial=True, fix_final=False)
    phase.add_state('y', fix_initial=True, fix_final=False)

    # Note that by omitting the targets here Dymos will automatically attempt to connect
    # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
    phase.add_state('v', fix_initial=True, fix_final=False)

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', targets=['g'], units='m/s**2')

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.setup()

    phase.set_simulate_options(method='RK23')

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 2.0

    p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
    p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
    p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
    p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
    p['traj0.phase0.parameters:g'] = 9.80665

    if do_reports:
        dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'reports': True})
    else:
        dm.run_problem(p, run_driver=True, simulate=True)

    return p


def setup_model_shooting(do_reports):
    prob = om.Problem()

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.declare_coloring(tol=1.0E-12)

    tx = dm.ExplicitShooting(grid=dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=6, compressed=False),
                             subprob_reports=do_reports)

    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

    phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

    # automatically discover states
    phase.set_state_options('x', fix_initial=True)
    phase.set_state_options('y', fix_initial=True)
    phase.set_state_options('v', fix_initial=True)

    phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
    phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9,
                      ref=90., rate2_continuity=True)

    phase.add_boundary_constraint('x', loc='final', equals=10.0)
    phase.add_boundary_constraint('y', loc='final', equals=5.0)

    prob.model.add_subsystem('phase0', phase)

    phase.add_objective('time', loc='final')

    prob.setup(force_alloc_complex=True)

    prob.set_val('phase0.t_initial', 0.0)
    prob.set_val('phase0.t_duration', 2)
    prob.set_val('phase0.states:x', 0.0)
    prob.set_val('phase0.states:y', 10.0)
    prob.set_val('phase0.states:v', 1.0E-6)
    prob.set_val('phase0.parameters:g', 1.0, units='m/s**2')
    prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

    dm.run_problem(prob, run_driver=True, simulate=False)

    return prob


# reports API between 3.18 and 3.19, so handle it here in order to be able to test against older
# versions of openmdao
if Version(openmdao_version) > Version("3.18"):
    from openmdao.utils.reports_system import get_reports_dir, clear_reports

    @use_tempdirs
    class TestSubproblemReportToggle(unittest.TestCase):

        def setUp(self):
            self.n2_filename = _default_n2_filename
            self.scaling_filename = _default_scaling_filename

            # set things to a known initial state for all the test runs
            openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs
            os.environ.pop('OPENMDAO_REPORTS', None)
            os.environ.pop('OPENMDAO_REPORTS_DIR', None)
            # We need to remove the TESTFLO_RUNNING environment variable for these tests to run.
            # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything if set
            # But we need to remember whether it was set so we can restore it
            self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
            clear_reports()

            self.count = 0

        def tearDown(self):
            # restore what was there before running the test
            if self.testflo_running is not None:
                os.environ['TESTFLO_RUNNING'] = self.testflo_running

        @hooks_active
        def test_no_sim_reports(self):
            p = setup_model_radau(do_reports=False)

            report_subdirs = sorted([e for e in pathlib.Path(get_reports_dir()).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            self.assertEqual(len(report_subdirs), 1)

            path = pathlib.Path(report_subdirs[0]).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(report_subdirs[0]).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

        @hooks_active
        def test_make_sim_reports(self):
            p = setup_model_radau(do_reports=True)

            report_subdirs = sorted([e for e in pathlib.Path(get_reports_dir()).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            # There is the nominal problem, the simulation problem, and a subproblem for the simulation.
            self.assertEqual(len(report_subdirs), 3)

            for subdir in report_subdirs:
                path = pathlib.Path(subdir).joinpath(self.n2_filename)
                self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')

        @hooks_active
        def test_explicitshooting_no_subprob_reports(self):
            p = setup_model_shooting(do_reports=False)

            report_subdirs = sorted([e for e in pathlib.Path(get_reports_dir()).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            self.assertEqual(len(report_subdirs), 1)

            path = pathlib.Path(report_subdirs[0]).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(report_subdirs[0]).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

        @hooks_active
        def test_explicitshooting_make_subprob_reports(self):
            p = setup_model_shooting(do_reports=True)

            report_subdirs = sorted([e for e in pathlib.Path(get_reports_dir()).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            # There is the nominal problem, a subproblem for integration, and a subproblem for the derivatives.
            self.assertEqual(len(report_subdirs), 2)

            path = pathlib.Path(report_subdirs[0]).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(report_subdirs[0]).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

            for subdir in report_subdirs:
                path = pathlib.Path(subdir).joinpath(self.n2_filename)
                self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')

else:  # old OM versions before reports API changed...
    from openmdao.utils.reports_system import set_default_reports_dir, _reports_dir, clear_reports, \
        setup_default_reports

    @use_tempdirs
    class TestSubproblemReportToggle(unittest.TestCase):

        def setUp(self):
            self.n2_filename = _default_n2_filename
            self.scaling_filename = _default_scaling_filename

            # set things to a known initial state for all the test runs
            openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs
            os.environ.pop('OPENMDAO_REPORTS', None)
            os.environ.pop('OPENMDAO_REPORTS_DIR', None)
            # We need to remove the TESTFLO_RUNNING environment variable for these tests to run.
            # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything if set
            # But we need to remember whether it was set so we can restore it
            self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
            clear_reports()
            set_default_reports_dir(_reports_dir)

            self.count = 0

        def tearDown(self):
            # restore what was there before running the test
            if self.testflo_running is not None:
                os.environ['TESTFLO_RUNNING'] = self.testflo_running

        @hooks_active
        def test_no_sim_reports(self):
            setup_default_reports()

            p = setup_model_radau(do_reports=False)

            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(p._name)
            report_subdirs = sorted([e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            self.assertEqual(len(report_subdirs), 1)

            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

        @hooks_active
        def test_make_sim_reports(self):
            setup_default_reports()

            p = setup_model_radau(do_reports=True)

            report_subdirs = sorted([e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            # # There is the nominal problem, the simulation problem, and a subproblem for each segment in the simulation.
            self.assertEqual(len(report_subdirs), 12)

            for subdir in report_subdirs:
                path = pathlib.Path(subdir).joinpath(self.n2_filename)
                self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')

        @hooks_active
        def test_explicitshooting_no_subprob_reports(self):
            setup_default_reports()

            p = setup_model_shooting(do_reports=False)

            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(p._name)
            report_subdirs = sorted([e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            self.assertEqual(len(report_subdirs), 1)

            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

        @hooks_active
        def test_explicitshooting_make_subprob_reports(self):
            setup_default_reports()

            p = setup_model_shooting(do_reports=True)

            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(p._name)
            report_subdirs = sorted([e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()])

            # Test that a report subdir was made
            # There is the nominal problem and a subproblem for integration
            self.assertEqual(len(report_subdirs), 2)

            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

            for subdir in report_subdirs:
                path = pathlib.Path(subdir).joinpath(self.n2_filename)
                self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
