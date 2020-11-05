import numpy as np
import openmdao.api as om
import dymos as dm

import unittest
from openmdao.utils.assert_utils import assert_near_equal


class ODEComp(om.ExplicitComponent):

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


class TestCannonballMatrixState(unittest.TestCase):
    """ Tests to verify that dymos can use matrix-states"""
    def _make_problem(self, tx):
        p = om.Problem()

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=ODEComp,
                         transcription=tx)
        traj.add_phase('phase', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(1, 5))
        phase.add_state('z', rate_source='zdot', fix_initial=True, units=None)

        phase.add_boundary_constraint('z', loc='final', lower=0, upper=0, indices=[1])
        phase.add_objective('time', loc='final')

        p.driver = om.pyOptSparseDriver()

        p.setup()

        p.set_val('traj.phase.t_initial', 0)
        p.set_val('traj.phase.t_duration', 5)
        p.set_val('traj.phase.states:z', phase.interpolate(ys=[[[0, 0], [10, 10]], [[10, 0], [10, -10]]], nodes='state_input'))

        return p

    def test_cannonball_matrix_state_radau(self):

        tx = dm.Radau(num_segments=10, order=3, solve_segments=True)

        p = self._make_problem(tx)

        p.run_driver()

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.03873598, tolerance=1E-5)
        assert_near_equal(p.get_val('traj.phase.timeseries.states:z')[-1, 0, 1], 0.0, tolerance=1E-5)
        assert_near_equal(p.get_val('traj.phase.timeseries.states:z')[-1, 0, 0], 20.3873598, tolerance=1E-5)

        p.model.traj.simulate()

    def test_cannonball_matrix_state_gl(self):
        tx = dm.GaussLobatto(num_segments=10, order=3, solve_segments=True)

        p = self._make_problem(tx)

        p.run_driver()

        assert_near_equal(p.get_val('traj.phase.timeseries.time')[-1], 2.03873598, tolerance=1E-5)
        assert_near_equal(p.get_val('traj.phase.timeseries.states:z')[-1, 0, 1], 0.0, tolerance=1E-5)
        assert_near_equal(p.get_val('traj.phase.timeseries.states:z')[-1, 0, 0], 20.3873598, tolerance=1E-5)