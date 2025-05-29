from re import M
import openmdao.api as om
import dymos as dm
import numpy as np

import unittest
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


class ODEComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('h', shape=nn, units='m')
        self.add_input('v', shape=nn, units='m/s')
        self.add_output('hdot', shape=nn, units='m/s')
        self.add_output('vdot', shape=nn, units='m/s**2')

        self.declare_partials(of='hdot', wrt='v', rows=np.arange(nn), cols=np.arange(nn), val=1.0)

    def compute(self, inputs, outputs):
        outputs['hdot'] = inputs['v']
        outputs['vdot'] = -9.80665


class MatrixStateCannonball(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # z is the state vector, a nn x 2 x 2 in the form of [[x, y], [vx, vy]]
        self.add_input('z', shape=(nn, 2, 2), units=None)
        self.add_output('zdot', shape=(nn, 2, 2), units=None)

        self.declare_partials(of='zdot', wrt='z', method='cs')
        self.declare_coloring(wrt=['z'], method='cs', num_full_jacs=5, tol=1.0E-12)

    def compute(self, inputs, outputs):
        outputs['zdot'][:, 0, 0] = inputs['z'][:, 1, 0]
        outputs['zdot'][:, 0, 1] = inputs['z'][:, 1, 1]
        outputs['zdot'][:, 1, 0] = 0.0
        outputs['zdot'][:, 1, 1] = -9.81


@use_tempdirs
class TestImplicitDuration(unittest.TestCase):

    @staticmethod
    def _make_simple_problem(tx, input_duration=False, fix_duration=False, expr_end=False):
        p = om.Problem()

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=ODEComp, transcription=tx)
        traj.add_phase('phase', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, units='s',
                               input_duration=input_duration, fix_duration=fix_duration)
        phase.add_state('h', rate_source='hdot', fix_initial=True, units='m', solve_segments='forward')
        phase.add_state('v', rate_source='vdot', fix_initial=True, units='m/s', solve_segments='forward')

        if not expr_end:
            phase.set_duration_balance('h', val=0.0, units='m')
        else:
            phase.set_duration_balance('pe=9.80665*h', val=0.0, units='m**2/s**2')

        phase.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        phase.linear_solver = om.DirectSolver()

        phase.set_simulate_options(rtol=1.0E-9, atol=1.0E-9)

        p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        p.driver.declare_coloring(tol=1.0E-12)

        phase.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        phase.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0, duration=2)
        phase.set_state_val('h', [30, 0])
        phase.set_state_val('v', [0, -10])

        return p

    @staticmethod
    def _make_matrix_problem(tx, shape_error=False):
        p = om.Problem()

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=MatrixStateCannonball, transcription=tx)
        traj.add_phase('phase', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, units='s')
        phase.add_state('z', rate_source='zdot', fix_initial=True, solve_segments='forward')

        index = None if shape_error else [[0], [1]]
        phase.set_duration_balance('z', val=0.0, index=index)

        phase.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        phase.linear_solver = om.DirectSolver()

        phase.set_simulate_options(rtol=1.0E-9, atol=1.0E-9)

        phase.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        phase.linear_solver = om.DirectSolver()

        p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        p.driver.declare_coloring(tol=1.0E-12)

        p.setup()

        p.set_val('traj.phase.t_initial', 0)
        p.set_val('traj.phase.t_duration', 7)
        p.set_val('traj.phase.states:z', phase.interp('z', [[[0, 0], [10, 10]], [[10, 0], [10, -10]]]))

        return p

    def test_implicit_duration_radau(self):
        tx = dm.Radau(num_segments=12, order=3, solve_segments=False)

        p = self._make_simple_problem(tx)

        dm.run_problem(p, run_driver=False, simulate=False)

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.4735192, tolerance=1E-6)
        assert_near_equal(p.get_val('traj.phase.timeseries.h')[-1], 0.0, tolerance=1E-6)

    def test_implicit_duration_gl(self):
        tx = dm.GaussLobatto(num_segments=12, order=3, solve_segments=False)

        p = self._make_simple_problem(tx)

        dm.run_problem(p, run_driver=False, simulate=True, make_plots=True)

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.4735192, tolerance=1E-6)
        assert_near_equal(p.get_val('traj.phase.timeseries.h')[-1], 0.0, tolerance=1E-6)

    def test_implicit_duration_shooting(self):
        tx = dm.ExplicitShooting(num_segments=12, order=3)

        p = self._make_simple_problem(tx)

        dm.run_problem(p, run_driver=False, simulate=False)

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.4735192, tolerance=1E-6)
        assert_near_equal(p.get_val('traj.phase.timeseries.h')[-1], 0.0, tolerance=1E-6)

    def test_implicit_duration_matrix_state(self):
        tx = dm.Radau(num_segments=12, order=3, solve_segments=False)

        p = self._make_matrix_problem(tx, shape_error=False)

        dm.run_problem(p, run_driver=False, simulate=False)

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.03873598, tolerance=1E-5)

    @unittest.skip("set_duration_balance/add_boundary_balance doesn't currently work with expressions.")
    @require_pyoptsparse(optimizer='IPOPT')
    def test_implicit_duration_radau_expr_condition(self):
        tx = dm.Radau(num_segments=12, order=3, solve_segments=False)

        p = self._make_simple_problem(tx, expr_end=True)

        dm.run_problem(p, run_driver=False, simulate=False)

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.4735192, tolerance=1E-6)
        assert_near_equal(p.get_val('traj.phase.timeseries.h')[-1], 0.0, tolerance=1E-6)

        print((p.get_val('traj.phase.timeseries.v')[-1]) ** 2 / 2)

    @unittest.skip("set_duration_balance/add_boundary_balance doesn't currently work with expressions.")
    def test_implicit_duration_gl_expr_condition(self):
        tx = dm.GaussLobatto(num_segments=12, order=3, solve_segments=False)

        p = self._make_simple_problem(tx, expr_end=True)

        dm.run_problem(p, run_driver=False, simulate=False)

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.4735192, tolerance=1E-6)
        assert_near_equal(p.get_val('traj.phase.timeseries.h')[-1], 0.0, tolerance=1E-6)


if __name__ == '__main__':
    unittest.main()
