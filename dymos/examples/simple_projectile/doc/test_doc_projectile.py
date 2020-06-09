import unittest
from dymos.utils.doc_utils import save_for_docs


class TestDocProjectile(unittest.TestCase):

    @save_for_docs
    def test_ivp(self):
        import openmdao.api as om
        import dymos as dm
        import matplotlib.pyplot as plt
        # plt.switch_backend('Agg')

        from projectile_ode import ProjectileODE

        prob = om.Problem()

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=ProjectileODE, transcription=dm.Radau(num_segments=10))

        phase.add_state('x', rate_source='x_dot', targets=None, units='m')
        phase.add_state('y', rate_source='y_dot', targets=None, units='m')
        phase.add_state('vx', rate_source='vx_dot', targets=['vx'], units='m/s')
        phase.add_state('vy', rate_source='vy_dot', targets=['vy'], units='m/s')

        traj.add_phase('phase0', phase)

        prob.model.add_subsystem('traj', traj)

        prob.setup()

        prob.set_val('traj.phase0.t_initial', 0.0)
        prob.set_val('traj.phase0.t_duration', 15.0)

        prob.set_val('traj.phase0.states:x', 0.0)
        prob.set_val('traj.phase0.states:y', 0.0)
        prob.set_val('traj.phase0.states:vx', 100.0)
        prob.set_val('traj.phase0.states:vy', 100.0)

        prob.run_model()

        exp_out = traj.simulate()

        x_exp = exp_out.get_val('traj.phase0.timeseries.states:x')
        y_exp = exp_out.get_val('traj.phase0.timeseries.states:y')

        plt.plot(x_exp, y_exp)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        plt.show()


if __name__ == '__main__':
    unittest.main()
