import unittest

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    plt.style.use('ggplot')
except ImportError:
    matplotlib = None

from openmdao.utils.testing_utils import use_tempdirs
from dymos.examples.robertson_problem.robertson_ode import RobertsonODE


@use_tempdirs
class TestRobertsonProblemForDocs(unittest.TestCase):

    def robertson_problem(self, t_final=1.0):

        import openmdao.api as om
        import dymos as dm

        #
        # Initialize the Problem
        #
        p = om.Problem(model=om.Group())

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=RobertsonODE,
                                        transcription=dm.ExplicitShooting(num_segments=10, method='LSODA')
                                        ))
        #
        # Set the variables
        #
        phase.set_time_options(fix_initial=True, fix_duration=True)

        phase.add_state('x0', fix_initial=True, fix_final=False, rate_source='xdot', targets='x')
        phase.add_state('y0', fix_initial=True, fix_final=False, rate_source='ydot', targets='y')
        phase.add_state('z0', fix_initial=True, fix_final=False, rate_source='zdot', targets='z')

        #
        # Setup the Problem
        #
        p.setup(check=True)

        #
        # Set the initial values
        #
        phase.set_time_val(initial=0.0, duration=t_final)
        phase.set_state_val('x0', [1.0, 0.7])
        phase.set_state_val('y0', [0.0, 1e-5])
        phase.set_state_val('z0', [0.0, 0.3])

        return p

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_robertson_problem_for_docs(self):

        import openmdao.api as om
        from dymos.utils.testing_utils import assert_check_partials
        from openmdao.utils.assert_utils import assert_near_equal
        import matplotlib.pyplot as plt

        num_nodes = 3

        p = om.Problem(model=om.Group())

        p.model.add_subsystem('ode', RobertsonODE(num_nodes=num_nodes), promotes=['*'])

        p.setup(force_alloc_complex=True)

        p.set_val('x', [10., 100., 1000.])
        p.set_val('y', [1, 0.1, 0.01])
        p.set_val('z', [1., 2., 3.])

        p.run_model()
        cpd = p.check_partials(method='cs', compact_print=True)

        assert_check_partials(cpd)

        assert_near_equal(p.get_val('xdot'), [9999.6, 1996., 260.])
        assert_near_equal(p.get_val('ydot'), [-3.00099996E7, -3.01996E5, -3.26E3])
        assert_near_equal(p.get_val('zdot'), [3.0E7, 3.0E5, 3.0E3])

        # just set up the problem, test it elsewhere
        p = self.robertson_problem(t_final=40)

        p.run_model()

        assert_near_equal(p.get_val('traj.phase0.timeseries.x0')[-1], 0.71583161, tolerance=1E-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.y0')[-1], 9.18571144e-06, tolerance=1E-4)
        assert_near_equal(p.get_val('traj.phase0.timeseries.z0')[-1], 0.2841592, tolerance=1E-4)

        t_sim = p.get_val('traj.phase0.timeseries.time')

        states = ['x0', 'y0', 'z0']
        fig, axes = plt.subplots(len(states), 1)
        for i, state in enumerate(states):
            axes[i].plot(t_sim, p.get_val(f'traj.phase0.timeseries.{state}'), 'o')
            axes[i].set_ylabel(state[0])
        axes[-1].set_xlabel('time (s)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
