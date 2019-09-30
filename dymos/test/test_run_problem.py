import unittest
import openmdao.api as om
import dymos as dm

from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestRunProblem(unittest.TestCase):

    def test_run_HS_problem(self):
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

    def test_run_brachistochrone_problem(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.opt_settings['iSumm'] = 6

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=10, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                         units=BrachistochroneODE.states['x']['units'],
                         fix_initial=True, fix_final=False, solve_segments=False)
        phase0.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                         units=BrachistochroneODE.states['y']['units'],
                         fix_initial=True, fix_final=False, solve_segments=False)
        phase0.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                         targets=BrachistochroneODE.states['v']['targets'],
                         units=BrachistochroneODE.states['v']['units'],
                         fix_initial=True, fix_final=False, solve_segments=False)
        phase0.add_control('theta', targets=BrachistochroneODE.parameters['theta']['targets'],
                           continuity=True, rate_continuity=True,
                           units='deg', lower=0.01, upper=179.9)
        phase0.add_input_parameter('g', targets=BrachistochroneODE.parameters['g']['targets'],
                                   units='m/s**2', val=9.80665)

        phase0.add_boundary_constraint('x', loc='final', equals=10)
        phase0.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase0.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[0, 10], nodes='state_input'))
        p.set_val('traj.phase0.states:y', phase0.interpolate(ys=[10, 5], nodes='state_input'))
        p.set_val('traj.phase0.states:v', phase0.interpolate(ys=[0, 9.9], nodes='state_input'))
        p.set_val('traj.phase0.controls:theta', phase0.interpolate(ys=[5, 100], nodes='control_input'))
        p.set_val('traj.phase0.input_parameters:g', 9.80665)

        dm.run_problem(p, True)
