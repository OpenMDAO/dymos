from __future__ import print_function, division, absolute_import
import unittest

from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.hull_problem.hull_ode import HullProblemODE

c = 5


@use_tempdirs
class TestHull(unittest.TestCase):

    @staticmethod
    @require_pyoptsparse(optimizer='IPOPT')
    def make_problem(transcription):
        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['mu_init'] = 1e-3
        p.driver.opt_settings['tol'] = 1e-8
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        p.driver.opt_settings['print_level'] = 0
        p.driver.declare_coloring()

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase', dm.Phase(ode_class=HullProblemODE,
                                                  transcription=transcription))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='u')
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L')
        phase0.add_control('u', opt=True, targets=['u'], ref0=-0.15, ref=-0.14)

        phase0.add_objective(f'J = {c}*x**2/2 + xL')

        p.setup(check=True)

        phase0.set_state_val('x', [1.5, 0.0])
        phase0.set_state_val('xL', [0.0, 0.0])
        phase0.set_time_val(initial=0.0, duration=10.0)
        phase0.set_control_val('u', [0.0, -0.14])
        return p

    @staticmethod
    def solution(x0, td):

        xf = x0 - c * x0 * td / (1 + c * td)
        uf = -c * x0 / (1 + c * td)

        return xf, uf

    def test_hull_gauss_lobatto(self):
        p = self.make_problem(transcription=dm.GaussLobatto(num_segments=20, order=3))
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_radau(self):
        p = self.make_problem(transcription=dm.Radau(num_segments=20, order=3))
        dm.run_problem(p, run_driver=True, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_birkhoff(self):
        p = self.make_problem(transcription=dm.Birkhoff(order=9))
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_shooting(self):
        p = self.make_problem(transcription=dm.ExplicitShooting(num_segments=20, order=3))
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)


if __name__ == '__main__':
    unittest.main()
