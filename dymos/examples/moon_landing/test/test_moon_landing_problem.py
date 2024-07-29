import numpy as np
import openmdao.api as om
import dymos as dm
import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from dymos.examples.moon_landing import MoonLandingProblemODE


@use_tempdirs
class TestMoonLandingProblem(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    def make_problem(self, grid_type):
        self.p = om.Problem(model=om.Group())
        self.p.driver = om.pyOptSparseDriver()
        self.p.driver.declare_coloring()
        self.p.driver.options['optimizer'] = 'IPOPT'
        self.p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        self.p.driver.opt_settings['print_level'] = 0
        self.p.driver.opt_settings['linear_solver'] = 'mumps'
        self.p.driver.declare_coloring()

        t = dm.Birkhoff(num_nodes=20, grid_type=grid_type)

        traj = self.p.model.add_subsystem('traj', dm.Trajectory())
        phase = dm.Phase(ode_class=MoonLandingProblemODE,
                         transcription=t)

        phase.set_time_options(fix_initial=True, fix_duration=False)
        phase.add_state('h', fix_initial=True, rate_source='h_dot')
        phase.add_state('v', fix_initial=True, rate_source='v_dot')
        phase.add_state('m', fix_initial=True, lower=1e-3, rate_source='m_dot')
        phase.add_control('T', lower=0.0, upper=1.227)

        phase.add_boundary_constraint('h', loc='final', equals=0.0)
        phase.add_boundary_constraint('v', loc='final', equals=0.0)

        phase.add_objective('m', scaler=-1)
        phase.set_simulate_options(atol=1.0E-1, rtol=1.0E-2)

        traj.add_phase('phase', phase)

        self.p.setup(check=True, force_alloc_complex=True)

        phase.set_time_val(initial=0.0, duration=1.0)
        phase.set_state_val('h', [1.0, 0.0])
        phase.set_state_val('v', [-0.783, 0.0])
        phase.set_state_val('m', [1.0, 0.2])
        phase.set_control_val('T', [0.0, 1.227])

    def test_problem_lgl(self):

        with dm.options.temporary(include_check_partials=True):
            self.make_problem(grid_type='lgl')
            dm.run_problem(self.p, simulate=True, simulate_kwargs={'times_per_seg': 100},
                           make_plots=True)

            with np.printoptions(linewidth=1024):
                self.p.check_partials(compact_print=True, method='cs')

        h = self.p.get_val('traj.phase.timeseries.h')
        v = self.p.get_val('traj.phase.timeseries.v')
        m = self.p.get_val('traj.phase.timeseries.m')
        T = self.p.get_val('traj.phase.timeseries.T')

        assert_near_equal(T[0], 0.0, tolerance=1e-5)
        assert_near_equal(T[-1], 1.227, tolerance=1e-5)
        assert_near_equal(h[-1], 0.0, tolerance=1e-5)
        assert_near_equal(v[-1], 0.0, tolerance=1e-5)
        assert_near_equal(m[-1], 0.3953, tolerance=1e-3)

    def test_problem_cgl(self):
        self.make_problem(grid_type='cgl')
        dm.run_problem(self.p, make_plots=True)
        h = self.p.get_val('traj.phase.timeseries.h')
        v = self.p.get_val('traj.phase.timeseries.v')
        m = self.p.get_val('traj.phase.timeseries.m')
        T = self.p.get_val('traj.phase.timeseries.T')

        assert_near_equal(T[0], 0.0, tolerance=1e-5)
        assert_near_equal(T[-1], 1.227, tolerance=1e-5)
        assert_near_equal(h[-1], 0.0, tolerance=1e-5)
        assert_near_equal(v[-1], 0.0, tolerance=1e-5)
        assert_near_equal(m[-1], 0.3953, tolerance=1e-4)
