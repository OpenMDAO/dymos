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
        self.add_input('y0', shape=(1,), units='unitless', tags=['dymos.state_initial_target:y'])
        self.add_output('y', shape=(nn,), units='unitless', tags=['dymos.state_source:y'])

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
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('y0', shape=(1,), units='unitless', tags=['dymos.state_initial_target:y'])
        self.add_output('y', shape=(nn,), units='unitless', tags=['dymos.state_source:y'])
        self.declare_coloring(method='cs', tol=1.0E-12)

    def compute(self, inputs, outputs):
        t = inputs['t']
        y_0 = inputs['y0']
        y_1 = inputs['y1']
        c_2 = y0
        c_1 = -1 / 12 + 1/6
        outputs['y'] = x ** 4 / 12 - x ** 3 / 6 + c_1 * x + c_2


class TestAnalyticPhaseSimpleSystem(unittest.TestCase):

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

    # No integrated states in an analytic phase, they're just outputs!
    def test_solution(self):

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = dm.Phase(ode_class=SimpleIVPSolution,
                         transcription=dm.Analytic(grid='radau', num_segments=5, order=3))
        traj.add_phase('phase', phase)

        phase.set_time_options(units='s', targets=['t'], fix_initial=True, fix_duration=True)
        phase.add_state('y', input_initial=True, initial_targets=['y0'])

        p.setup()

        p.set_val('traj.phase.t_initial', 0.0, units='s')
        p.set_val('traj.phase.t_duration', 2.0, units='s')
        p.set_val('traj.phase.initial_states:y', 0.5, units='unitless')

        p.run_model()

        t = p.get_val('traj.phase.timeseries.time', units='s')
        y = p.get_val('traj.phase.timeseries.states:y', units='unitless')

        expected_y = t ** 2 + 2 * t + 1 - 0.5 * np.exp(t)

        assert_near_equal(y, expected_y)

    # Integration constants determined via parameters.
    def test_solution_params_only(self):

        p = om.Problem()
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        # Option A
        phase = dm.Phase(ode_class=SimpleIVPSolution,
                         transcription=dm.Analytic(grid='radau', num_segments=5, order=3))

        # Option B
        # phase = dm.AnalyticPhase(ode_class=SimpleIVPSolution, grid='radau', num_segments=5, order=3)

        traj.add_phase('phase', phase)

        phase.set_time_options(units='s', targets=['t'], fix_initial=True, fix_duration=True)
        phase.add_state('y')
        phase.add_parameter('y0', opt=False, units='unitless', static_target=True)

        p.setup()

        p.set_val('traj.phase.t_initial', 0.0, units='s')
        p.set_val('traj.phase.t_duration', 2.0, units='s')
        p.set_val('traj.phase.parameters:y0', 0.5, units='unitless')

        p.run_model()

        t = p.get_val('traj.phase.timeseries.time', units='s')
        y = p.get_val('traj.phase.timeseries.states:y', units='unitless')

        expected_y = t ** 2 + 2 * t + 1 - 0.5 * np.exp(t)

        assert_near_equal(y, expected_y)
