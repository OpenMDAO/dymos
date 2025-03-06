import os
import unittest
from numpy.testing import assert_almost_equal
import jax
import jax.numpy as jnp

import openmdao.api as om
import dymos as dm

from openmdao.utils.testing_utils import use_tempdirs


class GuidedBrachistochroneODE(om.JaxExplicitComponent):
    """
    The brachistochrone EOM assuming 
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # static parameters
        self.add_input('v', desc='speed of the bead on the wire', shape=(nn,), units='m/s')
        self.add_input('g', desc='gravitational acceleration', units='m/s**2')
        self.add_input('time_phase', desc='phase elapsed time', shape=(nn,), units='s')
        self.add_input('theta_rate', desc='guidance parameter', units='rad/s')

        self.add_output('xdot', desc='velocity component in x', shape=(nn,), units='m/s',
                        tags=['dymos.state_rate_source:x', 'dymos.state_units:m'])

        self.add_output('ydot', desc='velocity component in y', shape=(nn,), units='m/s',
                        tags=['dymos.state_rate_source:y', 'dymos.state_units:m'])

        self.add_output('vdot', desc='acceleration magnitude', shape=(nn,), units='m/s**2',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])

        self.add_output('theta', desc='wire angle', shape=(nn,), val=2, units='rad')

    def compute_primal(self, v, g, time_phase, theta_rate):
        jax.debug.print('theta rate value is {k}', k=theta_rate)
        theta = time_phase * theta_rate

        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)

        vdot = g * cos_theta
        xdot = v * sin_theta
        ydot = -v * cos_theta

        return xdot, ydot, vdot, theta


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
        phase.add_parameter('theta_rate', units='rad/s')

        phase.add_timeseries_output('theta', units='deg')

        phase.add_boundary_balance(param='t_duration', name='x', tgt_val=10.0, loc='final', lower=0.1, upper=5.0)
        # phase.add_boundary_balance(param='parameters:theta_rate', name='y', tgt_val=5.0, loc='final', lower=0.1, upper=10.0, res_ref=1.0)
        
        phase.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0, atol=1.0E-3, rtol=1.0E-3)
        phase.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        phase.nonlinear_solver.debug_print = True
        phase.linear_solver = om.DirectSolver()

        bal = om.BalanceComp()
        bal = p.model.add_subsystem('theta_rate_balance', bal)
        bal.add_balance('parameters:theta_rate', lhs_name='y_final', rhs_val=5.0, lower=0.1, upper=10, units='rad/s', eq_units='m')
        p.model.connect('theta_rate_balance.parameters:theta_rate', 'traj0.phase0.parameters:theta_rate')
        p.model.connect('traj0.phase0.final_states:y', 'theta_rate_balance.y_final')

        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0, atol=1.0E-3, rtol=1.0E-3)
        p.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        p.model.nonlinear_solver.debug_print = True
        p.model.linear_solver = om.DirectSolver()

        p.set_solver_print(0)

        p.setup()

        phase.set_time_val(initial=0.0, duration=1.8016)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [1.0E-6, 9.9])

        phase.set_parameter_val('theta_rate',  100/1.8016)
        phase.set_parameter_val('g', 9.80665)

        p.final_setup()
        om.n2(p)

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

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(tf, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

        outputs = [op[1]['prom_name'] for op in p.model.list_outputs(out_stream=None, prom_name=True)]
        self.assertNotIn('traj0.phase0.timeseries.theta_rate', outputs)

    def test_picard_cgl(self):
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=11, grid_type='cgl')
        p = self.make_problem(tx=tx)

        # p.final_setup()

        p.run_model()

        # p.check_totals(of=['traj0.phase0.t_duration', 'traj0.phase0.parameters:theta_rate'],
        #                wrt=['traj0.phase0.states:x', 'traj0.phase0.states:y'])

        # om.n2(p)

        import matplotlib.pyplot as plt

        plt.switch_backend('macosx')
        fig, axes = plt.subplots(1, 2)
        t = p.get_val('traj0.phase0.timeseries.time')
        g = p.get_val('traj0.phase0.parameters:g')
        theta_rate = p.get_val('traj0.phase0.parameters:theta_rate')
        x = p.get_val('traj0.phase0.timeseries.x')
        y = p.get_val('traj0.phase0.timeseries.y')
        v = p.get_val('traj0.phase0.timeseries.v')
        theta = p.get_val('traj0.phase0.timeseries.theta')

        print(y[-1, ...])

        # axes[0].plot(x, y)
        # axes[1].plot(t, theta)

        # plt.show()


        # plt.switch_backend('macosx')
        # fig, axes = plt.subplots(1, 2)

        # for k in jnp.linspace(5, 15, 16):

        #     p.set_val('traj0.phase0.parameters:k', k)

        #     p.run_model()

        #     t = p.get_val('traj0.phase0.timeseries.time')
        #     g = p.get_val('traj0.phase0.parameters:g')
        #     k = p.get_val('traj0.phase0.parameters:k')
        #     x = p.get_val('traj0.phase0.timeseries.x')
        #     y = p.get_val('traj0.phase0.timeseries.y')
        #     v = p.get_val('traj0.phase0.timeseries.v')
        #     theta = p.get_val('traj0.phase0.timeseries.theta')

        #     axes[0].plot(x, y)
        #     axes[1].plot(t, theta)
        # plt.show()
        
        # self.run_asserts(p)
        # self.tearDown()


if __name__ == '__main__':
    unittest.main()