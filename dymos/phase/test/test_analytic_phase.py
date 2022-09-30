import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal


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

    def test_invalid_add_state_options(self):
        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=11)

        invalid_state_options = [('opt', False),
                                 ('fix_initial', True),
                                 ('fix_final', True),
                                 ('initial_bounds', (0, 0)),
                                 ('final_bounds', (0, 0)),
                                 ('val', 0.0),
                                 # ('desc', 'Foo'),
                                 ('rate_source', 'foo.bar'),
                                 ('targets', ['y']),
                                 ('lower', 0.0),
                                 ('upper', 0.0),
                                 ('scaler', 1.0),
                                 ('adder', 0.0),
                                 ('ref0', 0.0),
                                 ('ref', 1.0),
                                 ('defect_scaler', 1.0),
                                 ('defect_ref', 1.0),
                                 # ('continuity', True),
                                 ('solve_segments', False),
                                 ('input_initial', False)]

        for opt, val in invalid_state_options:
            kwargs = {opt: val}

            with self.assertRaises(NotImplementedError) as e:
                phase.add_state('y', **kwargs)

            expected = f'States in AnalyticPhase are strictly outputs of the ODE solution system. Option `{opt}` is ' \
                       'not a valid option for states in AnalyticPhase.'

            self.assertEqual(expected, str(e.exception))

    def test_invalid_set_state_options(self):
        phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, num_nodes=11)
        phase.add_state('y')

        invalid_state_options = [('opt', False),
                                 ('fix_initial', True),
                                 ('fix_final', True),
                                 ('initial_bounds', (0, 0)),
                                 ('final_bounds', (0, 0)),
                                 ('val', 0.0),
                                 ('rate_source', 'foo.bar'),
                                 ('targets', ['y']),
                                 ('lower', 0.0),
                                 ('upper', 0.0),
                                 ('scaler', 1.0),
                                 ('adder', 0.0),
                                 ('ref0', 0.0),
                                 ('ref', 1.0),
                                 ('defect_scaler', 1.0),
                                 ('defect_ref', 1.0),
                                 ('solve_segments', False),
                                 ('input_initial', False)]

        for opt, val in invalid_state_options:
            kwargs = {opt: val}

            with self.assertRaises(NotImplementedError) as e:
                phase.set_state_options('y', **kwargs)

            expected = f'States in AnalyticPhase are strictly outputs of the ODE solution system. Option `{opt}` is ' \
                       'not a valid option for states in AnalyticPhase.'

            self.assertEqual(expected, str(e.exception))
