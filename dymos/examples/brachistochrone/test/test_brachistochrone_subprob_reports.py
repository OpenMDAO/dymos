"""Unit Tests for the code that does automatic report generation"""
import unittest
import os

import openmdao
import openmdao.api as om
import openmdao.core.problem
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.tests.test_hooks import hooks_active
from openmdao.visualization.n2_viewer.n2_viewer import _default_n2_filename
from openmdao.visualization.scaling_viewer.scaling_report import _default_scaling_filename


import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


def setup_model_radau(do_reports, probname):
    p = om.Problem(model=om.Group(), name=probname)

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

    phase.set_time_val(initial=0.0, duration=2.0)

    phase.set_state_val('x', [0, 10])
    phase.set_state_val('y', [10, 5])
    phase.set_state_val('v', [0, 9.9])

    phase.set_control_val('theta', [5, 100.5])
    phase.set_parameter_val('g', 9.80665)

    dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'reports': do_reports})

    return p


def setup_model_shooting(do_reports, probname):
    prob = om.Problem(name=probname)

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

    phase.set_time_val(initial=0.0, duration=2.0)

    phase.set_state_val('x', [0, 10])
    phase.set_state_val('y', [10, 5])
    phase.set_state_val('v', [0, 9.9])

    phase.set_control_val('theta', [0.01, 90], units='deg')
    phase.set_parameter_val('g', 1.0)

    dm.run_problem(prob, run_driver=True, simulate=True)

    return prob


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

        self.count = 0

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    @hooks_active
    def test_no_sim_reports(self):
        p = setup_model_radau(do_reports=False, probname='test_no_sim_reports')

        main_outputs_dir = p.get_outputs_dir()

        sim_outputs_dir = main_outputs_dir / 'traj0_simulation_out'
        sim_reports_dir = sim_outputs_dir / 'reports'

        self.assertFalse(sim_reports_dir.exists())

    @hooks_active
    def test_make_sim_reports(self):
        p = setup_model_radau(do_reports=True, probname='test_make_sim_reports')

        main_reports_dir = p.get_reports_dir()

        traj = p.model._get_subsystem('traj0')
        sim_reports_dir = traj.sim_prob.get_reports_dir()

        self.assertTrue((main_reports_dir / self.n2_filename).exists())
        self.assertTrue(sim_reports_dir.exists())
        self.assertTrue((sim_reports_dir / self.n2_filename).exists())

    @hooks_active
    def test_explicitshooting_no_subprob_reports(self):
        p = setup_model_shooting(do_reports=False,
                                 probname='test_explicitshooting_no_subprob_reports')

        main_reports_dir = p.get_reports_dir()
        subprob_reports_dir = p.model.phase0.integrator._eval_subprob.get_reports_dir()

        main_reports = os.listdir(main_reports_dir)

        self.assertFalse(subprob_reports_dir.exists())

        self.assertIn(self.n2_filename, main_reports)
        self.assertIn(self.scaling_filename, main_reports)

    @hooks_active
    def test_explicitshooting_make_subprob_reports(self):
        p = setup_model_shooting(do_reports=True,
                                 probname='test_explicitshooting_make_subprob_reports')

        main_reports_dir = p.get_reports_dir()
        subprob_reports_dir = p.model.phase0.integrator._eval_subprob.get_reports_dir()

        main_reports = os.listdir(main_reports_dir)
        subprob_reports = os.listdir(subprob_reports_dir)

        self.assertIn(self.n2_filename, main_reports)
        self.assertIn(self.n2_filename, subprob_reports)

        self.assertIn(self.scaling_filename, main_reports)

        # The subprob has no optimization, so should not have a scaling report
        self.assertNotIn(self.scaling_filename, subprob_reports)


if __name__ == '__main__':
    unittest.main()
