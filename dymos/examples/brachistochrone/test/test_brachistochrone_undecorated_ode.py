from __future__ import print_function, division, absolute_import

import unittest
import numpy as np
from openmdao.api import ExplicitComponent


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


class TestBrachistochroneUndecoratedODE(unittest.TestCase):

    def test_brachistochrone_undecorated_ode_gl(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10), units='s')

        phase.set_state_options('x', fix_initial=True, fix_final=True, rate_source='xdot', units='m')
        phase.set_state_options('y', fix_initial=True, fix_final=True, rate_source='ydot', units='m')
        phase.set_state_options('v', fix_initial=True, rate_source='vdot', targets=['v'], units='m/s')

        phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9, targets=['theta'])

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # # Generate the explicitly simulated trajectory
        # t0 = p['phase0.t_initial']
        # tf = t0 + p['phase0.t_duration']
        # exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)
        #
        # fig, ax = plt.subplots()
        # fig.suptitle('Brachistochrone Solution')
        #
        # x_imp = p.get_val('phase0.timeseries.states:x')
        # y_imp = p.get_val('phase0.timeseries.states:y')
        #
        # x_exp = exp_out.get_val('phase0.timeseries.states:x')
        # y_exp = exp_out.get_val('phase0.timeseries.states:y')
        #
        # ax.plot(x_imp, y_imp, 'ro', label='solution')
        # ax.plot(x_exp, y_exp, 'b-', label='simulated')
        #
        # ax.set_xlabel('x (m)')
        # ax.set_ylabel('y (m)')
        # ax.grid(True)
        # ax.legend(loc='upper right')
        #
        # plt.show()

    def test_brachistochrone_undecorated_ode_radau(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('radau-ps',
                      ode_class=BrachistochroneODE,
                      num_segments=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10), units='s')

        phase.set_state_options('x', fix_initial=True, fix_final=True, rate_source='xdot', units='m')
        phase.set_state_options('y', fix_initial=True, fix_final=True, rate_source='ydot', units='m')
        phase.set_state_options('v', fix_initial=True, rate_source='vdot', targets=['v'], units='m/s')

        phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9, targets=['theta'])

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # # Generate the explicitly simulated trajectory
        # t0 = p['phase0.t_initial']
        # tf = t0 + p['phase0.t_duration']
        # exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)
        #
        # fig, ax = plt.subplots()
        # fig.suptitle('Brachistochrone Solution')
        #
        # x_imp = p.get_val('phase0.timeseries.states:x')
        # y_imp = p.get_val('phase0.timeseries.states:y')
        #
        # x_exp = exp_out.get_val('phase0.timeseries.states:x')
        # y_exp = exp_out.get_val('phase0.timeseries.states:y')
        #
        # ax.plot(x_imp, y_imp, 'ro', label='solution')
        # ax.plot(x_exp, y_exp, 'b-', label='simulated')
        #
        # ax.set_xlabel('x (m)')
        # ax.set_ylabel('y (m)')
        # ax.grid(True)
        # ax.legend(loc='upper right')
        #
        # plt.show()

    def test_brachistochrone_undecorated_ode_rk(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import RungeKuttaPhase

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = RungeKuttaPhase(ode_class=BrachistochroneODE, num_segments=20)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10), units='s')

        phase.set_state_options('x', fix_initial=True, rate_source='xdot', units='m')
        phase.set_state_options('y', fix_initial=True, rate_source='ydot', units='m')
        phase.set_state_options('v', fix_initial=True, rate_source='vdot', targets=['v'], units='m/s')

        phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9, targets=['theta'])

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # # Generate the explicitly simulated trajectory
        # t0 = p['phase0.t_initial']
        # tf = t0 + p['phase0.t_duration']
        # exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)
        #
        # fig, ax = plt.subplots()
        # fig.suptitle('Brachistochrone Solution')
        #
        # x_imp = p.get_val('phase0.timeseries.states:x')
        # y_imp = p.get_val('phase0.timeseries.states:y')
        #
        # x_exp = exp_out.get_val('phase0.timeseries.states:x')
        # y_exp = exp_out.get_val('phase0.timeseries.states:y')
        #
        # ax.plot(x_imp, y_imp, 'ro', label='solution')
        # ax.plot(x_exp, y_exp, 'b-', label='simulated')
        #
        # ax.set_xlabel('x (m)')
        # ax.set_ylabel('y (m)')
        # ax.grid(True)
        # ax.legend(loc='upper right')
        #
        # plt.show()


class TestBrachistochroneBasePhaseClass(unittest.TestCase):

    def test_brachistochrone_base_phase_class_gl(self):
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import GaussLobattoPhase

        class BrachistochronePhase(GaussLobattoPhase):

            def setup(self):

                self.options['ode_class'] = BrachistochroneODE

                self.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10), units='s')
                self.set_state_options('x', fix_initial=True, rate_source='xdot', units='m')
                self.set_state_options('y', fix_initial=True, rate_source='ydot', units='m')
                self.set_state_options('v', fix_initial=True, rate_source='vdot', targets=['v'], units='m/s')
                self.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9, targets=['theta'])
                self.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665, targets=['g'])

                super(BrachistochronePhase, self).setup()


        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = BrachistochronePhase(num_segments=20, transcription_order=3)
        p.model.add_subsystem('phase0', phase)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)
