import unittest

import numpy as np
import scipy.special
from numpy.testing import assert_almost_equal

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.pseudospectral.components.birkhoff_collocation_comp import BirkhoffCollocationComp
from dymos.transcriptions.grid_data import BirkhoffGrid

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
CollocationComp = CompWrapperConfig(BirkhoffCollocationComp)


class SimpleODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s**2')
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('p', shape=(nn,), units='s**2')

        self.add_output('x_dot', shape=(nn,), units='s')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='x_dot', wrt='p', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        t = inputs['t']
        p = inputs['p']
        outputs['x_dot'] = x - t**2 + p

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['x_dot', 't'] = -2*t


class TestCollocationComp(unittest.TestCase):

    def make_problem(self, grid_type='lgl'):
        dm.options['include_check_partials'] = True

        self.p = om.Problem(model=om.Group())

        gd = BirkhoffGrid(num_segments=1, nodes_per_seg=21, grid_type=grid_type)
        tx = dm.Birkhoff(grid=gd)
        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=SimpleODE, transcription=tx)
        self.p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, fix_duration=True)

        phase.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot')
        phase.add_control('t', opt=False)
        phase.add_parameter('p', targets=['p'], units='s**2')

        self.p.setup(force_alloc_complex=True)
        times = gd.node_stau + 1
        x_sol = times**2 + 2*times + 1 - 0.5*np.exp(times)
        x_dot_sol = x_sol - times**2 + 1

        self.p['traj0.phase0.t_initial'] = 0.0
        self.p['traj0.phase0.t_duration'] = 2.0

        self.p['traj0.phase0.states:x'] = phase.interp('x', ys=x_sol, xs=times)
        self.p['traj0.phase0.state_rates:x'] = phase.interp('x', ys=x_dot_sol, xs=times)
        self.p['traj0.phase0.controls:t'] = phase.interp('t', ys=times, xs=times)
        self.p['traj0.phase0.parameters:p'] = 1
        self.p['traj0.phase0.initial_states:x'] = x_sol[0]
        self.p['traj0.phase0.final_states:x'] = x_sol[-1]
        self.p['traj0.phase0.state_segment_ends:x'] = [x_sol[0], x_sol[-1]]

        self.p.run_model()

    def assert_results(self):
        assert_almost_equal(self.p['traj0.phase0.state_defects:x'], 0.0)
        assert_almost_equal(self.p['traj0.phase0.state_rate_defects:x'], 0.0)
        assert_almost_equal(self.p['traj0.phase0.initial_state_defects:x'], 0.0)
        assert_almost_equal(self.p['traj0.phase0.final_state_defects:x'], 0.0)

    def test_results_lgl_grid(self):
        self.make_problem(grid_type='lgl')
        self.assert_results()

    def test_results_cgl_grid(self):
        self.make_problem(grid_type='cgl')
        self.assert_results()

    def test_partials_lgl(self):
        self.make_problem(grid_type='lgl')
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='cs')
        assert_check_partials(cpd)

    def test_partials_cgl(self):
        self.make_problem(grid_type='cgl')
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
