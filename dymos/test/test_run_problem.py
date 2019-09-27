import unittest
import openmdao.api as om
import dymos as dm
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE


class TestRunProblem(unittest.TestCase):

    def test_run_problem(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.opt_settings['iSumm'] = 6

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE,
                                                transcription=dm.Radau(num_segments=10, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
        phase0.add_control('u', opt=True, targets=['u'], rate_continuity=False)

        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        p.setup(check=True)

        p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[1.5, 1], nodes='state_input'))
        p.set_val('traj.phase0.states:xL', phase0.interpolate(ys=[0, 1], nodes='state_input'))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', 100)
        p.set_val('traj.phase0.controls:u', phase0.interpolate(ys=[-0.6, 2.4],
                                                               nodes='control_input'))
        dm.run_problem(p, True)

