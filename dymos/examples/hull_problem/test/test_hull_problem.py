from __future__ import print_function, division, absolute_import
import unittest

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos import Trajectory, GaussLobatto, Phase, Radau, ExplicitShooting, PicardShooting
from dymos.examples.hull_problem.hull_ode import HullProblemODE

c = 5


@use_tempdirs
class TestHull(unittest.TestCase):

    @staticmethod
    @require_pyoptsparse(optimizer='IPOPT')
    def make_problem(transcription, control_cnty=True,
                     control_rate_cnty=True, control_rate2_cnty=False):
        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['tol'] = 1.0E-6
        p.driver.opt_settings['linear_solver'] = 'mumps'
        p.driver.declare_coloring()

        traj = p.model.add_subsystem('traj', Trajectory())
        phase0 = traj.add_phase('phase', Phase(ode_class=HullProblemODE,
                                               transcription=transcription))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='u')
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L')
        phase0.add_control('u', opt=True, targets=['u'],
                           continuity=control_cnty,
                           rate_continuity=control_rate_cnty,
                           rate2_continuity=control_rate2_cnty)

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
        p = self.make_problem(transcription=GaussLobatto(num_segments=30, order=3))
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_radau(self):
        p = self.make_problem(transcription=Radau(num_segments=30, order=3))
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_shooting(self):
        p = self.make_problem(transcription=ExplicitShooting(num_segments=30, order=3))
        dm.run_problem(p, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

    def test_hull_picard(self):
        p = self.make_problem(transcription=PicardShooting(num_segments=3, nodes_per_seg=11),
                              control_cnty=True,
                              control_rate_cnty=True,
                              control_rate2_cnty=False)
        dm.run_problem(p, run_driver=True, simulate=True)

        xf, uf = self.solution(1.5, 10)

        assert_near_equal(p.get_val('traj.phase.timeseries.x')[-1],
                          xf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase.continuity_comp.defect_controls:u'), np.zeros((2, 1)), tolerance=1.0E-4)
        assert_near_equal(p.get_val('traj.phase.continuity_comp.defect_control_rates:u_rate'), np.zeros((2, 1)), tolerance=1.0E-4)


if __name__ == '__main__':
    unittest.main()
