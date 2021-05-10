import os
import unittest

import numpy as np

import openmdao.api as om
from dymos.utils.testing_utils import assert_timeseries_near_equal
from openmdao.utils.testing_utils import use_tempdirs


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
class TestBrachistochroneIntegratedParameter(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_integrated_param_gauss_lobatto(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=True, rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, rate_source='vdot', units='m/s')
        phase.add_state('theta', fix_initial=False, rate_source='theta_dot', lower=1E-3)

        # theta_dot has no target, therefore we need to explicitly set the units and shape.
        phase.add_parameter('theta_dot', units='deg/s', shape=(1,), opt=True, lower=0, upper=100)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.states:theta'] = np.radians(phase.interp('theta', [0.05, 100.0]))
        p['phase0.parameters:theta_dot'] = 60.0

        # Solve for the optimal trajectory
        dm.run_problem(p, refine_iteration_limit=5)

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        sim_out = phase.simulate(times_per_seg=20)

        t_sol = p.get_val('phase0.timeseries.time')
        x_sol = p.get_val('phase0.timeseries.states:x')
        y_sol = p.get_val('phase0.timeseries.states:y')
        v_sol = p.get_val('phase0.timeseries.states:v')
        theta_sol = p.get_val('phase0.timeseries.states:theta')

        t_sim = sim_out.get_val('phase0.timeseries.time')
        x_sim = sim_out.get_val('phase0.timeseries.states:x')
        y_sim = sim_out.get_val('phase0.timeseries.states:y')
        v_sim = sim_out.get_val('phase0.timeseries.states:v')
        theta_sim = sim_out.get_val('phase0.timeseries.states:theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=5.0E-3)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=5.0E-3)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=5.0E-3)
        assert_timeseries_near_equal(t_sol, theta_sol, t_sim, theta_sim, tolerance=5.0E-3)

    def test_brachistochrone_integrated_parameter_radau_ps(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='xdot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=True, rate_source='ydot', units='m')
        phase.add_state('v', fix_initial=True, rate_source='vdot', units='m/s')
        phase.add_state('theta', fix_initial=False, rate_source='theta_dot', lower=1E-3)

        # theta_dot has no target, therefore we need to explicitly set the units and shape.
        phase.add_parameter('theta_dot', units='deg/s', shape=(1,), opt=True, lower=0, upper=100)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.states:theta'] = np.radians(phase.interp('theta', [0.05, 100.0]))
        p['phase0.parameters:theta_dot'] = 60.0

        # Solve for the optimal trajectory
        dm.run_problem(p, refine_iteration_limit=5)

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        sim_out = phase.simulate(times_per_seg=20)

        t_sol = p.get_val('phase0.timeseries.time')
        x_sol = p.get_val('phase0.timeseries.states:x')
        y_sol = p.get_val('phase0.timeseries.states:y')
        v_sol = p.get_val('phase0.timeseries.states:v')
        theta_sol = p.get_val('phase0.timeseries.states:theta')

        t_sim = sim_out.get_val('phase0.timeseries.time')
        x_sim = sim_out.get_val('phase0.timeseries.states:x')
        y_sim = sim_out.get_val('phase0.timeseries.states:y')
        v_sim = sim_out.get_val('phase0.timeseries.states:v')
        theta_sim = sim_out.get_val('phase0.timeseries.states:theta')

        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, tolerance=1.0E-3)
        assert_timeseries_near_equal(t_sol, theta_sol, t_sim, theta_sim, tolerance=1.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
