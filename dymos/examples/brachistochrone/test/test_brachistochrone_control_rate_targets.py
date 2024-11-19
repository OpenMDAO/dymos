import unittest

import numpy as np

try:
    import matplotlib
except ImportError:
    matplotlib = None

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from dymos.utils.misc import om_version
from dymos.utils.testing_utils import assert_timeseries_near_equal
from dymos.utils.misc import _unspecified

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
        self.options.declare('control_name', values=('theta', 'int_theta_rate'), default='int_theta_rate')

    def setup(self):
        nn = self.options['num_nodes']
        control_name = self.options['control_name']
        g_default_val = 9.80665 if self.options['static_gravity'] else 9.80665 * np.ones(nn)

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=g_default_val, desc='grav. acceleration', units='m/s/s')

        # Note, kind of strange for this to be named theta_rate, but this is just a demonstration that
        # we can connect to a control rate source in dymos.
        self.add_input(control_name, val=np.ones(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

        self.add_output('check', val=np.zeros(nn),
                        desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='vdot', wrt=control_name, rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt=control_name, rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt=control_name, rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt=control_name, rows=arange, cols=arange)

        if self.options['static_gravity']:
            c = np.zeros(self.options['num_nodes'])
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=c)
        else:
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        u_name = self.options['control_name']
        theta = inputs[u_name]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, partials):
        u_name = self.options['control_name']
        theta = inputs[u_name]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        partials['vdot', 'g'] = cos_theta
        partials['vdot', u_name] = -g * sin_theta

        partials['xdot', 'v'] = sin_theta
        partials['xdot', u_name] = v * cos_theta

        partials['ydot', 'v'] = -cos_theta
        partials['ydot', u_name] = v * sin_theta

        partials['check', 'v'] = 1 / sin_theta
        partials['check', u_name] = -v * cos_theta / sin_theta ** 2


# @use_tempdirs
class TestBrachistochroneControlRateTargets(unittest.TestCase):

    def test_brachistochrone_control_rate_targets(self):

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        transcriptions = {'gauss-lobatto': dm.GaussLobatto(num_segments=10),}
                         # 'radau': dm.Radau(num_segments=10),
                         # 'birkhoff': dm.Birkhoff(num_nodes=30)}

        for tx_name, tx in transcriptions.items():
            for control_target_method in ('implicit', 'explicit'):
                for control_type in ('full', 'polynomial'):

                    with self.subTest(f'{tx_name=} {control_target_method=} {control_type=}'):

                        p = om.Problem(model=om.Group())
                        p.driver = om.pyOptSparseDriver(optimizer='SLSQP', print_results=False)
                        p.driver.declare_coloring()

                        traj = dm.Trajectory()

                        ode_kwargs = {'control_name': 'int_theta_rate' if control_target_method == 'implicit' else 'theta'}

                        phase = dm.Phase(ode_class=BrachistochroneRateTargetODE,
                                        ode_init_kwargs=ode_kwargs,
                                        transcription=tx)

                        traj.add_phase('phase0', phase)

                        p.model.add_subsystem('traj0', traj)

                        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

                        phase.add_state('x', rate_source='xdot',
                                        units='m',
                                        fix_initial=True, fix_final=True, solve_segments=False)

                        phase.add_state('y', rate_source='ydot',
                                        units='m',
                                        fix_initial=True, fix_final=True, solve_segments=False)

                        phase.add_state('v', rate_source='vdot',
                                        units='m/s',
                                        fix_initial=True, fix_final=False, solve_segments=False)

                        # our control is the time integral of theta.  We pass its rate to the ODE and it should implicitly find it since
                        # it is named `int_theta_rate` in the ODE.
                        phase.add_control('int_theta', lower=0.0, upper=None, fix_initial=True,
                                        rate_targets=_unspecified if control_target_method == 'implicit' else ['theta'],
                                        control_type=control_type, order=7)

                        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

                        # Minimize time at the end of the phase
                        phase.add_objective('time', loc='final', scaler=10)

                        p.model.linear_solver = om.DirectSolver()

                        p.setup()

                        phase.set_time_val(initial=0.0, duration=2.0)

                        phase.set_state_val('x', [0, 10])
                        phase.set_state_val('y', [10, 5])
                        phase.set_state_val('v', [0, 9.9])
                        phase.set_control_val('int_theta', [0, 100], units='deg*s')

                        phase.timeseries_options['include_control_rates'] = True

                        p.run_model()

                        # Solve for the optimal trajectory
                        dm.run_problem(p, simulate=True, make_plots=True, simulate_kwargs={'reports': True})

                        sol_db = 'dymos_solution.db'
                        sim_db = 'dymos_simulation.db'
                        if om_version()[0] > (3, 34, 2):
                            sol_db = p.get_outputs_dir() / sol_db
                            sim_db = traj.sim_prob.get_outputs_dir() / sim_db

                        sol_case = om.CaseReader(sol_db).get_case('final')
                        sim_case = om.CaseReader(sim_db).get_case('final')

                        t_sol = sol_case.get_val('traj0.phase0.timeseries.time')
                        x_sol = sol_case.get_val('traj0.phase0.timeseries.x')
                        y_sol = sol_case.get_val('traj0.phase0.timeseries.y')
                        v_sol = sol_case.get_val('traj0.phase0.timeseries.v')
                        theta_sol = sol_case.get_val('traj0.phase0.timeseries.int_theta_rate')

                        t_sim = sim_case.get_val('traj0.phase0.timeseries.time')
                        x_sim = sim_case.get_val('traj0.phase0.timeseries.x')
                        y_sim = sim_case.get_val('traj0.phase0.timeseries.y')
                        v_sim = sim_case.get_val('traj0.phase0.timeseries.v')
                        theta_sim = sim_case.get_val('traj0.phase0.timeseries.int_theta_rate')

                        assert_timeseries_near_equal(t_sol, x_sol, t_sim, x_sim, rel_tolerance=5.0E-3, abs_tolerance=0.01)
                        assert_timeseries_near_equal(t_sol, y_sol, t_sim, y_sim, rel_tolerance=5.0E-3, abs_tolerance=0.01)
                        assert_timeseries_near_equal(t_sol, v_sol, t_sim, v_sim, rel_tolerance=5.0E-3, abs_tolerance=0.01)
                        assert_timeseries_near_equal(t_sol, theta_sol, t_sim, theta_sim, rel_tolerance=5.0E-3, abs_tolerance=0.01)


@use_tempdirs
class TestBrachistochronePolynomialControlRateTargets(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
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

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                          fix_initial=True, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [0, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
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

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                          fix_initial=True, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [0, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


@use_tempdirs
class TestBrachistochronePolynomialControlExplicitRateTargets(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
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

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                          rate_targets=['theta_rate'], fix_initial=True, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [0, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
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

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', order=3, units='deg*s', lower=0.01, upper=179.9,
                          rate_targets=['theta_rate'], fix_initial=True, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [0, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta_rate')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        theta_exp = exp_out.get_val('phase0.timeseries.theta_rate')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')
        ax.plot(t_exp, theta_exp, 'b-', label='simulated')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


@use_tempdirs
class TestBrachistochronePolynomialControlExplicitRate2Targets(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
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

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', order=5, units='deg*s**2', lower=0.01, upper=179.9,
                          rate_targets=None, rate2_targets=['theta_rate'], fix_initial=True, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [0, 10, 40, 60, 80, 100],
                              time_vals=[0, 0.4, 0.8, 1.2, 1.6, 2.0])

        # Solve for the optimal trajectory
        dm.run_problem(p, simulate=True)

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
