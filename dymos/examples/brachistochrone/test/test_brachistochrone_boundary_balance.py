import os
import unittest
from numpy.testing import assert_almost_equal

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    jax = None
    jnp = None

import openmdao.api as om
import dymos as dm

from openmdao.utils.testing_utils import use_tempdirs
from dymos.utils.misc import om_version


class GuidedBrachistochroneODE(om.JaxExplicitComponent):
    """
    The brachistochrone EOM assuming a linear rate of change in the wire angle wrt time.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # static parameters
        self.add_input('v', desc='speed of the bead on the wire', shape=(nn,), units='m/s')
        self.add_input('g', desc='gravitational acceleration', shape=(1,), units='m/s**2')
        self.add_input('time_phase', desc='phase elapsed time', shape=(nn,), units='s')
        self.add_input('theta_rate', desc='guidance parameter', shape=(1,), units='rad/s')

        self.add_output('xdot', desc='velocity component in x', shape=(nn,), units='m/s',
                        tags=['dymos.state_rate_source:x', 'dymos.state_units:m'])

        self.add_output('ydot', desc='velocity component in y', shape=(nn,), units='m/s',
                        tags=['dymos.state_rate_source:y', 'dymos.state_units:m'])

        self.add_output('vdot', desc='acceleration magnitude', shape=(nn,), units='m/s**2',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])

        self.add_output('theta', desc='wire angle', shape=(nn,), val=2, units='rad')

        # TODO: Temporary pending bug fix in JaxExplicitComp
        self.declare_coloring(method='jax')

    # because our compute primal output depends on static variables, in this case
    # and self.options['num_nodes'], we must define a get_self_statics method. This method must
    # return a tuple of all static variables. Their order in the tuple doesn't matter.  If your
    # component happens to have discrete inputs, do NOT return them here. Discrete inputs are passed
    # into the compute_primal function individually, after the continuous variables.
    def get_self_statics(self):
        # return value must be hashable
        return self.options['num_nodes'],

    def compute_primal(self, v, g, time_phase, theta_rate):
        theta = time_phase * theta_rate

        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        vdot = g * cos_theta
        xdot = v * sin_theta
        ydot = -v * cos_theta

        return xdot, ydot, vdot, theta


@unittest.skipIf(jax is None, 'Test Requires Jax')
@unittest.skipIf(om_version()[0] <= (3, 37, 0), 'Requires OpenMDAO version later than 3.37.0')
@use_tempdirs
class TestBrachistochroneBoundaryBalance(unittest.TestCase):

    def make_problem(self, tx):
        p = om.Problem(model=om.Group())

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=GuidedBrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_parameter('g', units='m/s**2')
        phase.add_parameter('theta_rate', units='deg/s')

        phase.add_calc_expr('v2 = v * v')

        phase.add_timeseries_output('theta', units='deg')
        phase.add_timeseries_output('v2', units='m**2/s**2')

        phase.add_boundary_balance(param='t_duration', name='x', tgt_val=10.0, loc='final', lower=0.1, upper=5.0)
        phase.add_boundary_balance(param='theta_rate', name='y', tgt_val=5.0, loc='final', lower=0.1, upper=100.0, res_ref=1.0)

        phase.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0, atol=1.0E-6, rtol=1.0E-6)
        phase.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        phase.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=1.8016)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_parameter_val('theta_rate',  20.0)
        phase.set_parameter_val('g', 9.80665)

        p.final_setup()

        return p

    def run_asserts(self, p):

        t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
        tf = p.get_val('traj0.phase0.timeseries.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.x')[0]
        xf = p.get_val('traj0.phase0.timeseries.x')[-1]

        y0 = p.get_val('traj0.phase0.timeseries.y')[0]
        yf = p.get_val('traj0.phase0.timeseries.y')[-1]

        v0 = p.get_val('traj0.phase0.timeseries.v')[0]
        vf = p.get_val('traj0.phase0.timeseries.v')[-1]

        g = p.get_val('traj0.phase0.parameter_vals:g')[0]

        thetaf = p.get_val('traj0.phase0.timeseries.theta')[-1]

        v2_f = p.get_val('traj0.phase0.timeseries.v2')[-1]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(tf, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(v2_f, vf**2, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

        outputs = [op[1]['prom_name'] for op in p.model.list_outputs(out_stream=None, prom_name=True)]
        self.assertNotIn('traj0.phase0.timeseries.theta_rate', outputs)

    def test_picard(self):
        for grid_type in 'cgl', 'lgl':
            with self.subTest(f'{grid_type=}'):
                tx = dm.PicardShooting(num_segments=1, nodes_per_seg=21, grid_type='cgl')
                p = self.make_problem(tx=tx)

                p.run_model()

                self.run_asserts(p)

    def test_birkhoff(self):
        for grid_type in 'cgl', 'lgl':
            with self.subTest(f'{grid_type=}'):
                tx = dm.Birkhoff(num_nodes=21, grid_type=grid_type, solve_segments='forward')
                p = self.make_problem(tx=tx)

                p.run_model()

                self.run_asserts(p)

    def test_radau(self):
        tx = dm.Radau(num_segments=10, order=3, solve_segments='forward')
        p = self.make_problem(tx=tx)

        p.run_model()

        self.run_asserts(p)

    def test_gausslobatto(self):
        tx = dm.GaussLobatto(num_segments=10, order=3, solve_segments='forward')
        p = self.make_problem(tx=tx)

        p.run_model()

        self.run_asserts(p)

    def test_explicit_shooting(self):
        tx = dm.ExplicitShooting(num_segments=10, order=3)
        p = self.make_problem(tx=tx)

        p.run_model()

        self.run_asserts(p)


if __name__ == '__main__':
    unittest.main()
