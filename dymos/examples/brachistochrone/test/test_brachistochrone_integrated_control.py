import os
import unittest

import numpy as np
from scipy.interpolate import interp1d

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


class BrachistochroneODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=9.80665 * np.ones(nn), desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        jacobian['vdot', 'g'] = cos_theta
        jacobian['vdot', 'theta'] = -g * sin_theta

        jacobian['xdot', 'v'] = sin_theta
        jacobian['xdot', 'theta'] = v * cos_theta

        jacobian['ydot', 'v'] = -cos_theta
        jacobian['ydot', 'theta'] = v * sin_theta

        jacobian['check', 'v'] = 1 / sin_theta
        jacobian['check', 'theta'] = -v * cos_theta / sin_theta**2


@use_tempdirs
class TestBrachistochroneIntegratedControl(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_integrated_control_gauss_lobatto(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10), units='s')

        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=True, rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, rate_source='vdot', units='m/s')
        phase.add_state('theta', targets='theta', fix_initial=False, rate_source='theta_dot')

        phase.add_control('theta_dot', units='deg/s', rate_continuity=True, shape=(1, ), lower=0, upper=60)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.states:theta'] = np.radians(phase.interpolate(ys=[0.05, 100.0],
                                                                nodes='state_input'))
        p['phase0.controls:theta_dot'] = phase.interpolate(ys=[50, 50], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        sim_out = phase.simulate(times_per_seg=20)

        x_sol = p.get_val('phase0.timeseries.states:x')
        y_sol = p.get_val('phase0.timeseries.states:y')
        v_sol = p.get_val('phase0.timeseries.states:v')
        theta_sol = p.get_val('phase0.timeseries.states:theta')
        theta_dot_sol = p.get_val('phase0.timeseries.controls:theta_dot')
        time_sol = p.get_val('phase0.timeseries.time')

        x_sim = sim_out.get_val('phase0.timeseries.states:x')
        y_sim = sim_out.get_val('phase0.timeseries.states:y')
        v_sim = sim_out.get_val('phase0.timeseries.states:v')
        theta_sim = sim_out.get_val('phase0.timeseries.states:theta')
        theta_dot_sim = sim_out.get_val('phase0.timeseries.controls:theta_dot')
        time_sim = sim_out.get_val('phase0.timeseries.time')

        x_interp = interp1d(time_sim[:, 0], x_sim[:, 0])
        y_interp = interp1d(time_sim[:, 0], y_sim[:, 0])
        v_interp = interp1d(time_sim[:, 0], v_sim[:, 0])
        theta_interp = interp1d(time_sim[:, 0], theta_sim[:, 0])
        theta_dot_interp = interp1d(time_sim[:, 0], theta_dot_sim[:, 0])

        assert_near_equal(x_interp(time_sol), x_sol, tolerance=1.0E-5)
        assert_near_equal(y_interp(time_sol), y_sol, tolerance=1.0E-5)
        assert_near_equal(v_interp(time_sol), v_sol, tolerance=1.0E-5)
        assert_near_equal(theta_interp(time_sol), theta_sol, tolerance=1.0E-5)
        assert_near_equal(theta_dot_interp(time_sol), theta_dot_sol, tolerance=1.0E-5)

    def test_brachistochrone_integrated_control_radau_ps(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=dm.Radau(num_segments=10))

        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10), units='s')

        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=True, rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, rate_source='vdot', units='m/s')
        phase.add_state('theta', targets='theta', fix_initial=False, rate_source='theta_dot')

        phase.add_control('theta_dot', units='deg/s', rate_continuity=True, shape=(1, ), lower=0, upper=60)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj.phase0.states:theta'] = np.radians(phase.interpolate(ys=[0.05, 100.0], nodes='state_input'))
        p['traj.phase0.controls:theta_dot'] = phase.interpolate(ys=[50, 50], nodes='control_input')

        # Solve for the optimal trajectory
        dm.run_problem(p, simulate=True, make_plots=True)

        sol_case = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case = om.CaseReader('dymos_simulation.db').get_case('final')

        # Test the results
        assert_near_equal(sol_case.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        x_sol = sol_case.get_val('traj.phase0.timeseries.states:x')
        y_sol = sol_case.get_val('traj.phase0.timeseries.states:y')
        v_sol = sol_case.get_val('traj.phase0.timeseries.states:v')
        theta_sol = sol_case.get_val('traj.phase0.timeseries.states:theta')
        theta_dot_sol = sol_case.get_val('traj.phase0.timeseries.controls:theta_dot')
        time_sol = sol_case.get_val('traj.phase0.timeseries.time')

        x_sim = sim_case.get_val('traj.phase0.timeseries.states:x')
        y_sim = sim_case.get_val('traj.phase0.timeseries.states:y')
        v_sim = sim_case.get_val('traj.phase0.timeseries.states:v')
        theta_sim = sim_case.get_val('traj.phase0.timeseries.states:theta')
        theta_dot_sim = sim_case.get_val('traj.phase0.timeseries.controls:theta_dot')
        time_sim = sim_case.get_val('traj.phase0.timeseries.time')

        x_interp = interp1d(time_sim[:, 0], x_sim[:, 0])
        y_interp = interp1d(time_sim[:, 0], y_sim[:, 0])
        v_interp = interp1d(time_sim[:, 0], v_sim[:, 0])
        theta_interp = interp1d(time_sim[:, 0], theta_sim[:, 0])
        theta_dot_interp = interp1d(time_sim[:, 0], theta_dot_sim[:, 0])

        assert_near_equal(x_interp(time_sol), x_sol, tolerance=1.0E-4)
        assert_near_equal(y_interp(time_sol), y_sol, tolerance=1.0E-4)
        assert_near_equal(v_interp(time_sol), v_sol, tolerance=1.0E-4)
        assert_near_equal(theta_interp(time_sol), theta_sol, tolerance=1.0E-4)
        assert_near_equal(theta_dot_interp(time_sol), theta_dot_sol, tolerance=1.0E-4)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
