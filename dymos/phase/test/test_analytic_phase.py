import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


class SimpleIVPSolution(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('y0', shape=(1,), units='unitless', tags=['dymos.static_target'])
        self.add_output('y', shape=(nn,), units='unitless')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='y', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='y', wrt='y0', rows=ar, cols=np.zeros_like(ar))

    def compute(self, inputs, outputs):
        t = inputs['t']
        y0 = inputs['y0']
        outputs['y'] = t ** 2 + 2 * t + 1 - y0 * np.exp(t)

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        y0 = inputs['y0']
        partials['y', 't'] = 2 * t + 2 - y0 * np.exp(t)
        partials['y', 'y0'] = -np.exp(t)


class SimpleBVPSolution(om.ExplicitComponent):
    """
    A basic BVP ODE solution taken from
    https://math.libretexts.org/Bookshelves/Applied_Mathematics/Numerical_Methods_(Chasnov)/07%3A_Ordinary_Differential_Equations/7.01%3A_Examples_of_Analytical_Solutions

    Find the solution to

    /begin{align}
        -\frac{d^{2} y}{d x^{2}}= x (1 - x)
    /end{align}

    subject to

    /begin{align}
        y(0) = 0 \\
        y(1) = 0
    /end{align}
    """

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s')
        self.add_input('y0', shape=(1,), units='unitless', tags=['dymos.static_target'])
        self.add_input('y1', shape=(1,), units='unitless', tags=['dymos.static_target'])
        self.add_output('y', shape=(nn,), units='unitless', tags=['dymos.state_source:y'])
        self.declare_coloring(method='cs', tol=1.0E-12)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y_0 = inputs['y0']
        y_1 = inputs['y1']
        c_2 = y_0
        c_1 = y_1 - 1/12 + 1/6 - c_2
        outputs['y'] = x ** 4 / 12 - x ** 3 / 6 + c_1 * x + c_2


class TestAnalyticPhaseSimpleResults(unittest.TestCase):

    def test_simple_ivp_system(self):

        p = om.Problem()
        p.model.add_subsystem('simple_ivp_solution', SimpleIVPSolution(num_nodes=100), promotes=['*'])
        p.setup(force_alloc_complex=True)

        p.set_val('t', np.linspace(0, 2, 100))
        p.set_val('y0', 0.5)

        p.run_model()

        t = p.get_val('t')
        y = p.get_val('y')

        expected_y = t ** 2 + 2 * t + 1 - 0.5 * np.exp(t)

        assert_near_equal(y, expected_y)

        p.check_partials(compact_print=True, method='cs')

    def test_simple_ivp(self):

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)

        traj.add_phase('phase', phase)

        phase.set_time_options(units='s', targets=['t'], fix_initial=True, fix_duration=True)
        phase.add_state('y')
        phase.add_parameter('y0', opt=False)

        p.setup()

        p.set_val('traj.phase.t_initial', 0.0, units='s')
        p.set_val('traj.phase.t_duration', 2.0, units='s')
        p.set_val('traj.phase.parameters:y0', 0.5, units='unitless')

        p.run_model()

        t = p.get_val('traj.phase.timeseries.time', units='s')
        y = p.get_val('traj.phase.timeseries.states:y', units='unitless')

        expected = lambda x: x ** 2 + 2 * x + 1 - 0.5 * np.exp(x)

        assert_near_equal(y, expected(t))

    def test_simple_bvp(self):

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.AnalyticPhase(ode_class=SimpleBVPSolution, num_nodes=11)

        traj.add_phase('phase', phase)

        phase.set_time_options(units='s', targets=['x'], fix_initial=True, fix_duration=True,
                               initial_val=0.0, duration_val=1.0)
        phase.add_parameter('y0', opt=False, val=0.0)
        phase.add_parameter('y1', opt=False, val=0.0)

        p.setup()

        p.run_model()

        t = p.get_val('traj.phase.timeseries.time', units='s')
        y = p.get_val('traj.phase.timeseries.states:y', units='unitless')

        expected = lambda x: x * (1 - x) * (1 + x - x**2) / 12

        assert_near_equal(y, expected(t))

    def test_renamed_state(self):

        class SolutionWithRenamedState(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=(int,))

            def setup(self):
                nn = self.options['num_nodes']
                self.add_input('t', shape=(nn,), units='s')
                self.add_input('y0', shape=(1,), units='unitless', tags=['dymos.static_target'])
                self.add_output('y', shape=(nn,), units='unitless', tags=['dymos.state_source:foo'])

                ar = np.arange(nn, dtype=int)
                self.declare_partials(of='y', wrt='t', rows=ar, cols=ar)
                self.declare_partials(of='y', wrt='y0', rows=ar, cols=np.zeros_like(ar))

            def compute(self, inputs, outputs):
                t = inputs['t']
                y0 = inputs['y0']
                outputs['y'] = t ** 2 + 2 * t + 1 - y0 * np.exp(t)

            def compute_partials(self, inputs, partials):
                t = inputs['t']
                y0 = inputs['y0']
                partials['y', 't'] = 2 * t + 2 - y0 * np.exp(t)
                partials['y', 'y0'] = -np.exp(t)

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.AnalyticPhase(ode_class=SolutionWithRenamedState, num_nodes=10)

        traj.add_phase('phase', phase)

        phase.set_time_options(units='s', targets=['t'], fix_initial=True, fix_duration=True)
        phase.add_parameter('y0', opt=False)

        p.setup()

        p.set_val('traj.phase.t_initial', 0.0, units='s')
        p.set_val('traj.phase.t_duration', 2.0, units='s')
        p.set_val('traj.phase.parameters:y0', 0.5, units='unitless')

        p.run_model()

        t = p.get_val('traj.phase.timeseries.time', units='s')
        y = p.get_val('traj.phase.timeseries.states:foo', units='unitless')

        expected = lambda x: x ** 2 + 2 * x + 1 - 0.5 * np.exp(x)

        assert_near_equal(y, expected(t))


@use_tempdirs
class TestAnalyticPhaseInvalidOptions(unittest.TestCase):

    def test_add_control(self):
        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=11)

        with self.assertRaises(NotImplementedError) as e:
            phase.add_control('foo')

        self.assertEqual('AnalyticPhase does not support controls.', str(e.exception))

    def test_set_control_options(self):
        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=11)

        with self.assertRaises(NotImplementedError) as e:
            phase.set_control_options('foo', lower=0)

        self.assertEqual('AnalyticPhase does not support controls.', str(e.exception))

    def test_add_polynomial_control(self):
        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=11)

        with self.assertRaises(NotImplementedError) as e:
            phase.add_polynomial_control('foo', order=2)

        self.assertEqual('AnalyticPhase does not support polynomial controls.', str(e.exception))

    def test_set_polynomial_control_options(self):
        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=11)

        with self.assertRaises(NotImplementedError) as e:
            phase.set_polynomial_control_options('foo', lower=0)

        self.assertEqual('AnalyticPhase does not support polynomial controls.', str(e.exception))

    def test_timeseries_expr(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)

        traj.add_phase('phase', phase)

        phase.set_time_options(units='s', targets=['t'], fix_initial=True, fix_duration=True)
        phase.add_state('y')
        phase.add_parameter('y0', opt=False)
        phase.add_timeseries_output('z = y0 + y**2')

        p.setup()

        p.set_val('traj.phase.t_initial', 0.0, units='s')
        p.set_val('traj.phase.t_duration', 2.0, units='s')
        p.set_val('traj.phase.parameters:y0', 0.5, units='unitless')

        p.run_model()

        y = p.get_val('traj.phase.timeseries.states:y', units='unitless')
        y0 = p.get_val('traj.phase.timeseries.parameters:y0', units='unitless')

        expected_z = y0 + y**2
        z = p.get_val('traj.phase.timeseries.z')

        assert_near_equal(z, expected_z)


class TestLinkedAnalyticPhases(unittest.TestCase):

    def test_linked_phases_connected_false(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        first_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        first_phase.set_time_options(units='s', targets=['t'], fix_initial=True, duration_bounds=(0.5, 10.0))
        first_phase.add_state('y')
        first_phase.add_parameter('y0', opt=False)

        first_phase.add_boundary_constraint('y', loc='final', equals=1.5, units='unitless')

        second_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        second_phase.set_time_options(units='s', targets=['t'], fix_initial=False, duration_bounds=(0.1, 10.0))
        second_phase.add_state('y')
        second_phase.add_parameter('y0', opt=True)

        second_phase.add_boundary_constraint('time', loc='final', equals=2.0, units='s')

        # Since we're using constraints to enforce continuity between the two phases, we need a
        # driver and a dummy objective.  As usual, time works well for a dummy objective here.
        first_phase.add_objective('time', loc='final')
        p.driver = om.ScipyOptimizeDriver()

        traj.add_phase('first_phase', first_phase)
        traj.add_phase('second_phase', second_phase)

        traj.link_phases(['first_phase', 'second_phase'], ['time', 'y'], connected=False)

        p.setup()

        p.set_val('traj.first_phase.t_initial', 0.0, units='s')
        p.set_val('traj.first_phase.t_duration', 2.0, units='s')
        p.set_val('traj.first_phase.parameters:y0', 0.5, units='unitless')

        p.run_driver()

        t_1 = p.get_val('traj.first_phase.timeseries.time', units='s')
        x_1 = p.get_val('traj.first_phase.timeseries.states:y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.states:y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        # A dense version of the analytic solution for plot comparison.
        expected = lambda time: time ** 2 + 2 * time + 1 - y0_1 * np.exp(time)
        t_dense = np.linspace(t_1[0], t_2[-1], 100)

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)

    def test_linked_phases_connected_time(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        first_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        first_phase.set_time_options(units='s', targets=['t'], fix_initial=True, duration_bounds=(0.5, 10.0))
        first_phase.add_state('y')
        first_phase.add_parameter('y0', opt=False)

        first_phase.add_boundary_constraint('y', loc='final', equals=1.5, units='unitless')

        second_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        second_phase.set_time_options(units='s', targets=['t'], fix_initial=False, duration_bounds=(0.1, 10.0))
        second_phase.add_state('y')
        second_phase.add_parameter('y0', opt=True)

        second_phase.add_boundary_constraint('time', loc='final', equals=2.0, units='s')

        # Since we're using constraints to enforce continuity between the two phases, we need a
        # driver and a dummy objective.  As usual, time works well for a dummy objective here.
        first_phase.add_objective('time', loc='final')
        p.driver = om.ScipyOptimizeDriver()

        traj.add_phase('first_phase', first_phase)
        traj.add_phase('second_phase', second_phase)

        # We can link time with a connection, since initial time is an input to the second phase.
        traj.link_phases(['first_phase', 'second_phase'], ['time'], connected=True)
        traj.link_phases(['first_phase', 'second_phase'], ['y'], connected=False)

        p.setup()

        p.set_val('traj.first_phase.t_initial', 0.0, units='s')
        p.set_val('traj.first_phase.t_duration', 2.0, units='s')
        p.set_val('traj.first_phase.parameters:y0', 0.5, units='unitless')

        p.run_driver()

        t_1 = p.get_val('traj.first_phase.timeseries.time', units='s')
        x_1 = p.get_val('traj.first_phase.timeseries.states:y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.states:y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        # A dense version of the analytic solution for plot comparison.
        expected = lambda time: time ** 2 + 2 * time + 1 - y0_1 * np.exp(time)
        t_dense = np.linspace(t_1[0], t_2[-1], 100)

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)

    def test_linked_phases_connected_state(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        first_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        first_phase.set_time_options(units='s', targets=['t'], fix_initial=True, duration_bounds=(0.5, 10.0))
        first_phase.add_state('y')
        first_phase.add_parameter('y0', opt=False)

        first_phase.add_boundary_constraint('y', loc='final', equals=1.5, units='unitless')

        second_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        second_phase.set_time_options(units='s', targets=['t'], fix_initial=False, duration_bounds=(0.1, 10.0))
        second_phase.add_state('y')
        second_phase.add_parameter('y0', opt=True)

        second_phase.add_boundary_constraint('time', loc='final', equals=2.0, units='s')

        # Since we're using constraints to enforce continuity between the two phases, we need a
        # driver and a dummy objective.  As usual, time works well for a dummy objective here.
        first_phase.add_objective('time', loc='final')
        p.driver = om.ScipyOptimizeDriver()

        traj.add_phase('first_phase', first_phase)
        traj.add_phase('second_phase', second_phase)

        # We can link time with a connection, since initial time is an input to the second phase.
        traj.link_phases(['first_phase', 'second_phase'], ['time'], connected=True)
        traj.link_phases(['first_phase', 'second_phase'], ['y'], connected=True)

        with self.assertRaises(Exception) as e:
            p.setup()

        expected = "traj: Phase `first_phase` links variable `y` to phase `second_phase` state variable `y` by " \
                   "connection, but phase `second_phase` is an AnalyticPhase and does not support linking initial " \
                   "state values with option `connected=True`."

        self.assertEqual(expected, str(e.exception))

    def test_common_traj_param(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        first_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        first_phase.set_time_options(units='s', targets=['t'], fix_initial=True, duration_bounds=(0.5, 10.0))
        first_phase.add_state('y')
        first_phase.add_parameter('y0', opt=False)

        first_phase.add_boundary_constraint('y', loc='final', equals=1.5, units='unitless')

        second_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        second_phase.set_time_options(units='s', targets=['t'], fix_initial=False, duration_bounds=(0.1, 10.0))
        second_phase.add_state('y')
        second_phase.add_parameter('y0', opt=False)
        second_phase.add_boundary_constraint('time', loc='final', equals=2.0, units='s')

        # Since we're using constraints to enforce continuity between the two phases, we need a
        # driver and a dummy objective.  As usual, time works well for a dummy objective here.
        first_phase.add_objective('time', loc='final')
        p.driver = om.ScipyOptimizeDriver()

        traj.add_phase('first_phase', first_phase)
        traj.add_phase('second_phase', second_phase)

        # We can link time with a connection, since initial time is an input to the second phase.
        traj.link_phases(['first_phase', 'second_phase'], ['time'], connected=True)

        # Make the y0 parameter common between the two phases, since the particular solution to the ODE applies to both
        traj.add_parameter('y0', opt=False)

        p.setup()

        p.set_val('traj.first_phase.t_initial', 0.0, units='s')
        p.set_val('traj.first_phase.t_duration', 2.0, units='s')
        p.set_val('traj.parameters:y0', 0.5, units='unitless')

        p.run_driver()

        t_1 = p.get_val('traj.first_phase.timeseries.time', units='s')
        x_1 = p.get_val('traj.first_phase.timeseries.states:y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.states:y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        # A dense version of the analytic solution for plot comparison.
        expected = lambda time: time ** 2 + 2 * time + 1 - y0_1 * np.exp(time)
        t_dense = np.linspace(t_1[0], t_2[-1], 100)

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)

    def test_link_params_connected_false(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        first_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        first_phase.set_time_options(units='s', targets=['t'], fix_initial=True, duration_bounds=(0.5, 10.0))
        first_phase.add_state('y')
        first_phase.add_parameter('y0', opt=False)

        first_phase.add_boundary_constraint('y', loc='final', equals=1.5, units='unitless')

        second_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        second_phase.set_time_options(units='s', targets=['t'], fix_initial=False, duration_bounds=(0.1, 10.0))
        second_phase.add_state('y')
        second_phase.add_parameter('y0', opt=True)
        second_phase.add_boundary_constraint('time', loc='final', equals=2.0, units='s')

        # Since we're using constraints to enforce continuity between the two phases, we need a
        # driver and a dummy objective.  As usual, time works well for a dummy objective here.
        first_phase.add_objective('time', loc='final')
        p.driver = om.ScipyOptimizeDriver()

        traj.add_phase('first_phase', first_phase)
        traj.add_phase('second_phase', second_phase)

        # We can link time with a connection, since initial time is an input to the second phase.
        traj.link_phases(['first_phase', 'second_phase'], ['time'], connected=True)
        traj.link_phases(['first_phase', 'second_phase'], ['y0'], connected=False)

        p.setup()

        p.set_val('traj.first_phase.t_initial', 0.0, units='s')
        p.set_val('traj.first_phase.t_duration', 2.0, units='s')
        p.set_val('traj.first_phase.parameters:y0', 0.5, units='unitless')

        p.run_driver()

        t_1 = p.get_val('traj.first_phase.timeseries.time', units='s')
        x_1 = p.get_val('traj.first_phase.timeseries.states:y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.states:y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        # A dense version of the analytic solution for plot comparison.
        expected = lambda time: time ** 2 + 2 * time + 1 - y0_1 * np.exp(time)
        t_dense = np.linspace(t_1[0], t_2[-1], 100)

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)

    def test_link_params_connected_true(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        first_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        first_phase.set_time_options(units='s', targets=['t'], fix_initial=True, duration_bounds=(0.5, 10.0))
        first_phase.add_state('y')
        first_phase.add_parameter('y0', opt=False)

        first_phase.add_boundary_constraint('y', loc='final', equals=1.5, units='unitless')

        second_phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)
        second_phase.set_time_options(units='s', targets=['t'], fix_initial=False, duration_bounds=(0.1, 10.0))
        second_phase.add_state('y')
        second_phase.add_parameter('y0', opt=True)
        second_phase.add_boundary_constraint('time', loc='final', equals=2.0, units='s')

        # Since we're using constraints to enforce continuity between the two phases, we need a
        # driver and a dummy objective.  As usual, time works well for a dummy objective here.
        first_phase.add_objective('time', loc='final')
        p.driver = om.ScipyOptimizeDriver()

        traj.add_phase('first_phase', first_phase)
        traj.add_phase('second_phase', second_phase)

        # We can link time with a connection, since initial time is an input to the second phase.
        traj.link_phases(['first_phase', 'second_phase'], ['time'], connected=True)
        traj.link_phases(['first_phase', 'second_phase'], ['y0'], connected=True)

        p.setup()

        p.set_val('traj.first_phase.t_initial', 0.0, units='s')
        p.set_val('traj.first_phase.t_duration', 2.0, units='s')
        p.set_val('traj.first_phase.parameters:y0', 0.5, units='unitless')

        dm.run_problem(p, simulate=False)

        t_1 = p.get_val('traj.first_phase.timeseries.time', units='s')
        x_1 = p.get_val('traj.first_phase.timeseries.states:y', units='unitless')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.states:y', units='unitless')

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)

        with self.assertRaises(RuntimeError) as e:
            dm.run_problem(p, simulate=True)

        expected = 'Trajectory `traj` has no phases that support simulation.'
        self.assertEqual(expected, str(e.exception))

    def test_wtf(self):
        r = np.array([[2.43202063e-01, 9.54623741e-01, -7.56113684e-05],
               [2.44379694e-01, 9.54545890e-01, -2.14243165e-04],
               [3.51452372e-01, 9.48003731e-01, -6.53163638e-03],
               [4.30227289e-01, 9.62442889e-01, -2.84452628e-03],
               [4.85717379e-01, 9.20591861e-01, -4.79324018e-03],
               [5.37367866e-01, 8.71290201e-01, -6.74236017e-03],
               [5.51754163e-01, 7.60687681e-01, -1.49328403e-02],
               [6.66805115e-01, 7.23525394e-01, 2.74715457e-03],
               [7.75248767e-01, 6.79286865e-01, 2.02666957e-02],
               [8.76977622e-01, 6.29287873e-01, 3.74274994e-02],
               [9.72100657e-01, 5.74683851e-01, 5.40675338e-02],
               [1.17966034e+00, 6.27548504e-01, 1.40700823e-02],
               [1.38365557e+00, 6.78371909e-01, -2.61611041e-02],
               [1.76929347e+00, 7.38144101e-01, 2.78382890e-03],
               [2.22563925e+00, 8.04478727e-01, 5.92939759e-02],
               [2.63873113e+00, 8.55299550e-01, 9.58539933e-02],
               [3.05345867e+00, 9.06193121e-01, 1.31882329e-01],
               [3.47034845e+00, 9.57120238e-01, 1.67336167e-01],
               [3.88980487e+00, 1.00796872e+00, 2.02177798e-01],
               [4.55360118e+00, 1.00044648e+00, 2.07723705e-01]])

        x = r[:, 0]
        y = r[:, 1]
        z = r[:, 2]

        rmag_check = np.array([0.9851161 , 0.98533209, 1.01107493, 1.05422954, 1.04088122,
       1.02369737, 0.93984115, 0.98393374, 1.03094715, 1.08004342,
       1.13055938, 1.33626854, 1.54122535, 1.91709777, 2.36731322,
       2.7755409 , 3.18781881, 3.60380339, 4.02336411, 4.66683255])

        rmag = np.sqrt(x**2 + y**2 + z**2)

        phi = np.arccos(z / rmag)

        print(rmag - rmag_check)
        print(phi)