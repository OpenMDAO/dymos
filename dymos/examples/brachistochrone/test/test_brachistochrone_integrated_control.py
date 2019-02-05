from __future__ import print_function, absolute_import, division

import os
import unittest

import numpy as np

from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state, declare_parameter


@declare_time(units='s')
@declare_state('x', rate_source='xdot', units='m')
@declare_state('y', rate_source='ydot', units='m')
@declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
@declare_state('theta', targets=['theta'], rate_source='theta_dot', units='rad')
@declare_parameter('theta_dot', targets=[], units='rad/s')
@declare_parameter('g', units='m/s**2', targets=['g'])
class BrachistochroneODE(ExplicitComponent):

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


class TestBrachistochroneIntegratedControl(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_integrated_control_gauss_lobatto(self):
        import numpy as np
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)
        phase.set_state_options('theta', fix_initial=False)

        phase.add_control('theta_dot', units='deg/s', rate_continuity=True, lower=0, upper=60)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

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
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        phase.simulate(times=np.linspace(t0, tf, 50))

    def test_brachistochrone_integrated_control_radau_ps(self):
        import numpy as np
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('radau-ps',
                      ode_class=BrachistochroneODE,
                      num_segments=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)
        phase.set_state_options('theta', fix_initial=False)

        phase.add_control('theta_dot', units='deg/s', rate_continuity=True, lower=0, upper=60)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

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
        tf = p.get_val('phase0.time')[-1]
        assert_rel_error(self, tf, 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        phase.simulate(times=np.linspace(t0, tf, 50))

    def test_brachistochrone_integrated_control_single_shooting(self):
        import numpy as np
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, NonlinearBlockGS
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('explicit',
                      ode_class=BrachistochroneODE,
                      num_segments=5,
                      num_steps=10,
                      shooting='single',
                      seg_solver_class=NonlinearBlockGS)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)
        phase.set_state_options('theta', fix_initial=False)

        phase.add_control('theta_dot', units='deg/s', rate_continuity=True, lower=0, upper=60)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

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
        assert_rel_error(self, phase.get_values('time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        phase.simulate(times=np.linspace(t0, tf, 50))

    def test_brachistochrone_integrated_control_multiple_shooting(self):
        import numpy as np
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, NonlinearBlockGS
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('explicit',
                      ode_class=BrachistochroneODE,
                      num_segments=5,
                      num_steps=10,
                      shooting='multiple',
                      seg_solver_class=NonlinearBlockGS)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)
        phase.set_state_options('theta', fix_initial=False)

        phase.add_control('theta_dot', units='deg/s', rate_continuity=True, lower=0, upper=60)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

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
        assert_rel_error(self, phase.get_values('time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        phase.simulate(times=np.linspace(t0, tf, 50))
