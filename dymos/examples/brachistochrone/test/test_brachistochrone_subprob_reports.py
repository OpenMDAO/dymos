"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import sys
import os
from io import StringIO

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar_feature import SellarMDA
import openmdao.core.problem
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.reports_system import set_default_reports_dir, _reports_dir, register_report, \
    list_reports, clear_reports, run_n2_report, setup_default_reports, report_function
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI
from openmdao.utils.tests.test_hooks import hooks_active
from openmdao.visualization.n2_viewer.n2_viewer import _default_n2_filename
from openmdao.visualization.scaling_viewer.scaling_report import _default_scaling_filename

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


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

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
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

        dm.run_problem(p, run_driver=True, simulate=True)

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(p._name)
        report_subdirs = [e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()]

        # Test that a report subdir was made
        self.assertEqual(len(report_subdirs), 1)

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

    @hooks_active
    def test_make_sim_reports(self):
        setup_default_reports()

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

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
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

        dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'reports': True})

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(p._name)
        report_subdirs = [e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()]

        # Test that a report subdir was made
        # # There is the nominal problem, the simulation problem, and a subproblem for each segment in the simulation.
        self.assertEqual(len(report_subdirs), 12)

        for subdir in report_subdirs:
            path = pathlib.Path(subdir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')

    @hooks_active
    def test_explicitshooting_no_subprob_reports(self):
        setup_default_reports()

        prob = om.Problem()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.declare_coloring(tol=1.0E-12)

        tx = dm.ExplicitShooting(num_segments=3, grid='gauss-lobatto',
                                 method='rk4', order=5,
                                 num_steps_per_segment=5,
                                 compressed=False)

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

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)
        report_subdirs = [e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()]

        # Test that a report subdir was made
        self.assertEqual(len(report_subdirs), 1)

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

    @hooks_active
    def test_explicitshooting_make_subprob_reports(self):
        setup_default_reports()

        prob = om.Problem()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.declare_coloring(tol=1.0E-12)

        tx = dm.ExplicitShooting(num_segments=3, grid='gauss-lobatto',
                                 method='rk4', order=5,
                                 num_steps_per_segment=5,
                                 compressed=False,
                                 subprob_reports=True)

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

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)
        report_subdirs = [e for e in pathlib.Path(_reports_dir).iterdir() if e.is_dir()]

        # Test that a report subdir was made
        # There is the nominal problem, a subproblem for integration, and a subproblem for the derivatives.
        self.assertEqual(len(report_subdirs), 3)

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

        for subdir in report_subdirs:
            path = pathlib.Path(subdir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
