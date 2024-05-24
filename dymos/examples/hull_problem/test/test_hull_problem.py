from __future__ import print_function, division, absolute_import
import unittest

from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos import Trajectory, GaussLobatto, Phase, Radau, ExplicitShooting
from dymos.examples.hull_problem.hull_ode import HullProblemODE

c = 5


@use_tempdirs
class TestHull(unittest.TestCase):

    @staticmethod
    @require_pyoptsparse(optimizer='IPOPT')
    def make_problem(transcription=GaussLobatto, numseg=30):
        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['linear_solver'] = 'mumps'
        p.driver.declare_coloring()

        traj = p.model.add_subsystem('traj', Trajectory())
        phase0 = traj.add_phase('phase', Phase(ode_class=HullProblemODE,
                                               transcription=transcription(num_segments=numseg, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='u')
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L')
        phase0.add_control('u', opt=True, targets=['u'])

        phase0.add_objective(f'J = {c}*x**2/2 + xL')

        p.setup(check=True)

        phase0.set_state_val('x', [1.5, 1])
        phase0.set_state_val('xL', [0, 1])
        phase0.set_time_val(initial=0.0, duration=10.0)
        phase0.set_control_val('u', [-7, -0.14])
        return p

    @staticmethod
    def solution(x0, td):

        xf = x0 - c * x0 * td / (1 + c * td)
        uf = -c * x0 / (1 + c * td)

        return xf, uf

    def test_hull_gauss_lobatto(self):
        p = self.make_problem(transcription=GaussLobatto)
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_radau(self):
        p = self.make_problem(transcription=Radau)
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_shooting(self):
        p = self.make_problem(transcription=ExplicitShooting)
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)
