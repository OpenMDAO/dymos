import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_check_partials
from dymos.transcriptions.timestepping.euler_integration_comp import EulerIntegrationComp


class SimpleODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s**2')
        self.add_input('t', shape=(nn,), units='s')

        self.add_output('x_dot', shape=(nn,), units='s')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        x = inputs['x']
        t = inputs['t']
        outputs['x_dot'] = x - t**2 + 1

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['x_dot', 't'] = -2*t


class TestEulerIntegrationComp(unittest.TestCase):

    def test_fwd(self):
        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['targets'] = 't'
        time_options['units'] = 's'

        state_options = {'x': dm.phase.options.StateOptionsDictionary()}

        state_options['x']['shape'] = (1,)
        state_options['x']['units'] = 's**2'
        state_options['x']['rate_source'] = 'x_dot'
        state_options['x']['targets'] = ['x']

        control_options = {}
        polynomial_control_options = {}
        parameter_options = {}

        p = om.Problem()

        p.model.add_subsystem('fixed_step_integrator', EulerIntegrationComp(SimpleODE, time_options, state_options,
                                                                            parameter_options, control_options,
                                                                            polynomial_control_options, mode='fwd',
                                                                            num_steps=100,
                                                                            ode_init_kwargs=None))
        p.setup(mode='fwd', force_alloc_complex=True)

        p.set_val('fixed_step_integrator.state_initial_value:x', 0.5)
        p.set_val('fixed_step_integrator.t_initial', 0.0)
        p.set_val('fixed_step_integrator.t_duration', 2.0)

        p.run_model()

        cpd = p.check_partials(method='fd', form='central', compact_print=True)
        assert_check_partials(cpd)

    @unittest.skip('Not implemented')
    def test_rev(self):
        ode_class = SimpleODE
        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['targets'] = 't'
        time_options['units'] = 's'

        state_options = {'x': dm.phase.options.StateOptionsDictionary()}

        state_options['x']['shape'] = (1,)
        state_options['x']['units'] = 's**2'
        state_options['x']['rate_source'] = 'x_dot'
        state_options['x']['targets'] = ['x']

        control_options = {}
        polynomial_control_options = {}
        parameter_options = {}

        p = om.Problem()

        p.model.add_subsystem('fixed_step_integrator', EulerIntegrationComp(ode_class, time_options, state_options,
                                                                            parameter_options, control_options,
                                                                            polynomial_control_options, mode='rev',
                                                                            num_steps=10,
                                                                            ode_init_kwargs=None))
        p.setup(mode='rev', force_alloc_complex=True)

        p.set_val('fixed_step_integrator.state_initial_value:x', 0.5)
        p.set_val('fixed_step_integrator.t_initial', 0.0)
        p.set_val('fixed_step_integrator.t_duration', 2.0)

        p.run_model()

        cpd = p.check_partials(method='fd', form='central', compact_print=True)

        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
