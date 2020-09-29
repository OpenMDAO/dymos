import unittest
import openmdao.api as om
import numpy as np


class BrachistochroneRateTargetODE(om.ExplicitComponent):
    #
    # The following dictionaries provide a way of 'tagging' the Brachistochrone ODE with
    # information about states and parameters that can be accessed from Phase.
    #
    # In this case these are class attributes, but the choice of whether or not to tie
    # this information to the ODE itself (and how to do so) is entirely up to the user.
    #
    # In a dynamic ODE model these might be instance attributes whose values vary depending on
    # the arguments to the instantiation.
    #
    states = {'x': {'rate_source': 'xdot',
                    'units': 'm'},
              'y': {'rate_source': 'ydot',
                    'units': 'm'},
              'v': {'rate_source': 'vdot',
                    'units': 'm/s'}}

    parameters = {'theta': {'units': 'rad'},
                  'g': {'units': 'm/s**2'}}

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('static_gravity', types=(bool,), default=False,
                             desc='If True, treat gravity as a static (scalar) input, rather than '
                                  'having different values at each node.')

    def setup(self):
        nn = self.options['num_nodes']
        g_default_val = 9.80665 if self.options['static_gravity'] else 9.80665 * np.ones(nn)

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=g_default_val, desc='grav. acceleration', units='m/s/s')

        # Note, kind of strange for this to be named theta_rate, but this is just a demonstration that
        # we can connect to a control rate source in dymos.
        self.add_input('theta_rate', val=np.ones(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

        self.add_output('check', val=np.zeros(nn),
                        desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='vdot', wrt='theta_rate', rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta_rate', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta_rate', rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta_rate', rows=arange, cols=arange)

        if self.options['static_gravity']:
            c = np.zeros(self.options['num_nodes'])
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=c)
        else:
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta_rate']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, partials):
        theta = inputs['theta_rate']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        partials['vdot', 'g'] = cos_theta
        partials['vdot', 'theta_rate'] = -g * sin_theta

        partials['xdot', 'v'] = sin_theta
        partials['xdot', 'theta_rate'] = v * cos_theta

        partials['ydot', 'v'] = -cos_theta
        partials['ydot', 'theta_rate'] = v * sin_theta

        partials['check', 'v'] = 1 / sin_theta
        partials['check', 'theta_rate'] = -v * cos_theta / sin_theta ** 2


class TestBrachistochroneControlRateTargets(unittest.TestCase):

    def test_brachistochrone_control_rate_targets_gauss_lobatto(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', lower=0.01, upper=179.9, fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0, 100], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate')

        self.assertEqual(p.model.get_var_meta('phase0.timeseries.controls:theta', 'units'), 'rad*s')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_control_rate_targets_radau(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', units='deg*s', lower=0.01, upper=179.9, fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0, 100], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_control_rate_targets_rungekutta(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', units='deg*s', lower=0.01, upper=179.9, fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0, 100], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochroneExplicitControlRateTargets(unittest.TestCase):

    def test_brachistochrone_control_rate_targets_gauss_lobatto(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', units='deg*s', lower=0.01, upper=179.9, fix_initial=True,
                          rate_targets=['theta_rate'])

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0, 100], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_control_rate_targets_radau(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', units='deg*s', lower=0.01, upper=179.9, fix_initial=True,
                          rate_targets=['theta_rate'])

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0, 100], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_control_rate_targets_rungekutta(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', units='deg*s', lower=0.01, upper=179.9, fix_initial=True,
                          rate_targets=['theta_rate'])

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[0, 100], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlRateTargets(unittest.TestCase):

    def test_brachistochrone_polynomial_control_rate_targets_gauss_lobatto(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                                     fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rate_targets_radau(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                                     fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rate_targets_rungekutta(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                                     fix_initial=True, val=1.0)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlExplicitRateTargets(unittest.TestCase):

    def test_brachistochrone_polynomial_control_rate_targets_gauss_lobatto(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                                     rate_targets=['theta_rate'], fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rate_targets_radau(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                                     rate_targets=['theta_rate'], fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rate_targets_rungekutta(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                                     rate_targets=['theta_rate'], fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlExplicitRate2Targets(unittest.TestCase):

    def test_brachistochrone_polynomial_control_rate_targets_gauss_lobatto(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=5, units='deg*s**2', lower=0.01, upper=179.9,
                                     rate_targets=None, rate2_targets=['theta_rate'], fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 40, 60, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rate_targets_rungekutta(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                         transcription=dm.RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', order=5, units='deg*s**2', lower=0.01, upper=179.9,
                                     rate_targets=None, rate2_targets=['theta_rate'], fix_initial=True)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'] = [0, 10, 40, 60, 80, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_control_rates:theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
