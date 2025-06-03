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


@use_tempdirs
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

        p.check_partials(compact_print=True, method='cs', out_stream=None)

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

        p.check_partials(compact_print=True, method='cs', out_stream=None)

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
        y = p.get_val('traj.phase.timeseries.y', units='unitless')

        expected = t ** 2 + 2 * t + 1 - 0.5 * np.exp(t)

        assert_near_equal(y, expected)

    def test_simple_ivp_calc_expr(self):

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)

        phase.add_calc_expr('y2 = y**2')

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
        y = p.get_val('traj.phase.timeseries.y', units='unitless')
        y2 = p.get_val('traj.phase.timeseries.y2')

        expected = t ** 2 + 2 * t + 1 - 0.5 * np.exp(t)

        assert_near_equal(y, expected)
        assert_near_equal(y2, y*y)

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
        y = p.get_val('traj.phase.timeseries.y', units='unitless')

        expected = t * (1 - t) * (1 + t - t**2) / 12

        assert_near_equal(y, expected)

    def test_duplicate(self):

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.AnalyticPhase(ode_class=SimpleBVPSolution, num_nodes=11)

        phase.set_time_options(units='s', targets=['x'], fix_initial=True, fix_duration=True,
                               initial_val=0.0, duration_val=1.0)
        phase.add_parameter('y0', opt=False, val=0.0)
        phase.add_parameter('y1', opt=False, val=0.0)

        phase2 = phase.duplicate()
        traj.add_phase('phase2', phase2)

        p.setup()

        p.run_model()

        t = p.get_val('traj.phase2.timeseries.time', units='s')
        y = p.get_val('traj.phase2.timeseries.y', units='unitless')

        expected = t * (1 - t) * (1 + t - t**2) / 12

        assert_near_equal(y, expected)

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
        y = p.get_val('traj.phase.timeseries.foo', units='unitless')

        expected = t ** 2 + 2 * t + 1 - 0.5 * np.exp(t)

        assert_near_equal(y, expected)

    def test_analytic_phase_load_case(self):

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

        dm.run_problem(p, simulate=False)

        # Change the inputs so that we get different answers unless we reload the same case.
        p.set_val('traj.phase.t_initial', 0.0, units='s')
        p.set_val('traj.phase.t_duration', 1.0, units='s')
        p.set_val('traj.phase.parameters:y0', 0.6, units='unitless')

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'

        # Load the previous case and rerun
        dm.run_problem(p, simulate=False, restart=sol_db)

        t = p.get_val('traj.phase.timeseries.time', units='s')
        y = p.get_val('traj.phase.timeseries.y', units='unitless')

        expected = t ** 2 + 2 * t + 1 - 0.5 * np.exp(t)

        assert_near_equal(y, expected)


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

    def test_timeseries_expr(self):
        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=10)

        traj.add_phase('phase', phase)

        phase.set_time_options(units='s', targets=['t'], fix_initial=True, fix_duration=True)
        phase.add_state('y')
        phase.add_parameter('y0', units='unitless', opt=False)
        phase.add_timeseries_output('z = y0 + y**2', y0={'units': 'unitless'})

        p.setup()

        p.set_val('traj.phase.t_initial', 0.0, units='s')
        p.set_val('traj.phase.t_duration', 2.0, units='s')
        p.set_val('traj.phase.parameters:y0', 0.5, units='unitless')

        p.run_model()

        y = p.get_val('traj.phase.timeseries.y', units='unitless')
        y0 = p.get_val('traj.phase.parameter_vals:y0', units='unitless')

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
        x_1 = p.get_val('traj.first_phase.timeseries.y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(y0_1, y0_2, tolerance=1.0E-6)

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
        x_1 = p.get_val('traj.first_phase.timeseries.y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(y0_1, y0_2, tolerance=1.0E-6)

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
        x_1 = p.get_val('traj.first_phase.timeseries.y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(y0_1, y0_2, tolerance=1.0E-6)

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
        x_1 = p.get_val('traj.first_phase.timeseries.y', units='unitless')
        y0_1 = p.get_val('traj.first_phase.parameter_vals:y0')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.y', units='unitless')
        y0_2 = p.get_val('traj.second_phase.parameter_vals:y0')

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(y0_1, y0_2, tolerance=1.0E-6)

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
        x_1 = p.get_val('traj.first_phase.timeseries.y', units='unitless')

        t_2 = p.get_val('traj.second_phase.timeseries.time', units='s')
        x_2 = p.get_val('traj.second_phase.timeseries.y', units='unitless')

        assert_near_equal(1.500000, x_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(0.5338712554624387, t_1[-1, 0], tolerance=1.0E-6)
        assert_near_equal(2.0, t_2[-1, 0], tolerance=1.0E-6)
        assert_near_equal(5.305471950533106, x_2[-1, 0], tolerance=1.0E-6)

        with self.assertRaises(RuntimeError) as e:
            dm.run_problem(p, simulate=True)

        expected = 'Trajectory `traj` has no phases that support simulation.'
        self.assertEqual(expected, str(e.exception))
