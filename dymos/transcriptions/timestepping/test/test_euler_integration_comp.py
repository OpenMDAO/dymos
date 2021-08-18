import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from dymos.transcriptions.timestepping.euler_integration_comp import EulerIntegrationComp
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


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
        self.add_input('p', shape=(nn,), units='s**2')

        self.add_output('x_dot', shape=(nn,), units='s')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='x_dot', wrt='p', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        t = inputs['t']
        p = inputs['p']
        outputs['x_dot'] = x - t**2 + p

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['x_dot', 't'] = -2*t


class TestEulerIntegrationComp(unittest.TestCase):

    def test_eval_f_scalar(self):
        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['targets'] = 't'
        time_options['units'] = 's'

        state_options = {'x': dm.phase.options.StateOptionsDictionary()}

        state_options['x']['shape'] = (1,)
        state_options['x']['units'] = 's**2'
        state_options['x']['rate_source'] = 'x_dot'
        state_options['x']['targets'] = ['x']

        param_options = {'p': dm.phase.options.ParameterOptionsDictionary()}

        param_options['p']['shape'] = (1,)
        param_options['p']['units'] = 's**2'
        param_options['p']['targets'] = ['p']

        control_options = {}
        polynomial_control_options = {}

        prob = om.Problem()

        prob.model.add_subsystem('fixed_step_integrator',
                                 EulerIntegrationComp(SimpleODE, time_options, state_options,
                                                      param_options, control_options,
                                                      polynomial_control_options, mode='fwd',
                                                      num_steps=100, ode_init_kwargs=None))
        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.set_val('fixed_step_integrator.state_initial_values:x', 0.5)
        prob.set_val('fixed_step_integrator.t_initial', 0.0)
        prob.set_val('fixed_step_integrator.t_duration', 2.0)
        prob.set_val('fixed_step_integrator.parameters:p', 1.0)

        x = np.array([0.5])
        t = np.array([0.0])
        p = np.array([1.0])
        u = np.empty((0,))

        f = prob.model.fixed_step_integrator.eval_f(x, t, p, u)

        assert_near_equal(f, x - t**2 + p)

    def test_eval_f_derivs_scalar(self):
        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['targets'] = 't'
        time_options['units'] = 's'

        state_options = {'x': dm.phase.options.StateOptionsDictionary()}

        state_options['x']['shape'] = (1,)
        state_options['x']['units'] = 's**2'
        state_options['x']['rate_source'] = 'x_dot'
        state_options['x']['targets'] = ['x']

        param_options = {'p': dm.phase.options.ParameterOptionsDictionary()}

        param_options['p']['shape'] = (1,)
        param_options['p']['units'] = 's**2'
        param_options['p']['targets'] = ['p']

        control_options = {}
        polynomial_control_options = {}

        prob = om.Problem()

        prob.model.add_subsystem('fixed_step_integrator',
                                 EulerIntegrationComp(SimpleODE, time_options, state_options,
                                                      param_options, control_options,
                                                      polynomial_control_options, mode='fwd',
                                                      num_steps=100, complex_step_mode=True,
                                                      ode_init_kwargs=None))
        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.set_val('fixed_step_integrator.state_initial_values:x', 0.5)
        prob.set_val('fixed_step_integrator.t_initial', 0.0)
        prob.set_val('fixed_step_integrator.t_duration', 2.0)
        prob.set_val('fixed_step_integrator.parameters:p', 1.0)

        x = np.array([0.5], dtype=complex)
        t = np.array([0.0], dtype=complex)
        p = np.array([1.0], dtype=complex)
        u = np.empty((0,), dtype=complex)

        f_x, f_t, f_p, f_u, pu_pt = prob.model.fixed_step_integrator.eval_f_derivs(x, t, p, u)

        step = 1.0E-20

        f_x_cs = prob.model.fixed_step_integrator.eval_f(x + step * 1.0j, t, p, u).imag / step

        assert_near_equal(f_x.real, np.atleast_2d(f_x_cs))

        f_t_cs = prob.model.fixed_step_integrator.eval_f(x, t + step * 1.0j, p, u).imag / step

        assert_near_equal(f_t.real, np.atleast_2d(f_t_cs))

        f_p_cs = prob.model.fixed_step_integrator.eval_f(x, t, p + step * 1.0j, u).imag / step

        assert_near_equal(f_p.real, np.atleast_2d(f_p_cs))

    def test_fwd_parameters(self):
        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['targets'] = 't'
        time_options['units'] = 's'

        state_options = {'x': dm.phase.options.StateOptionsDictionary()}

        state_options['x']['shape'] = (1,)
        state_options['x']['units'] = 's**2'
        state_options['x']['rate_source'] = 'x_dot'
        state_options['x']['targets'] = ['x']

        param_options = {'p': dm.phase.options.ParameterOptionsDictionary()}

        param_options['p']['shape'] = (1,)
        param_options['p']['units'] = 's**2'
        param_options['p']['targets'] = ['p']

        control_options = {}
        polynomial_control_options = {}

        p = om.Problem()

        p.model.add_subsystem('fixed_step_integrator', EulerIntegrationComp(SimpleODE, time_options, state_options,
                                                                            param_options, control_options,
                                                                            polynomial_control_options, mode='fwd',
                                                                            num_steps=1000,
                                                                            ode_init_kwargs=None))
        p.setup(mode='fwd', force_alloc_complex=True)

        p.set_val('fixed_step_integrator.state_initial_values:x', 0.5)
        p.set_val('fixed_step_integrator.t_initial', 0.0)
        p.set_val('fixed_step_integrator.t_duration', 2.0)
        p.set_val('fixed_step_integrator.parameters:p', 1.0)

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

        p.set_val('fixed_step_integrator.state_initial_values:x', 0.5)
        p.set_val('fixed_step_integrator.t_initial', 0.0)
        p.set_val('fixed_step_integrator.t_duration', 2.0)

        p.run_model()

        cpd = p.check_partials(method='fd', form='central', compact_print=True)

        assert_check_partials(cpd)

    def test_fwd_parameters_controls(self):
        gd = dm.transcriptions.grid_data.GridData(num_segments=10, transcription='gauss-lobatto',
                                                  transcription_order=3)

        time_options = dm.phase.options.TimeOptionsDictionary()

        time_options['units'] = 's'

        state_options = {'x': dm.phase.options.StateOptionsDictionary(),
                         'y': dm.phase.options.StateOptionsDictionary(),
                         'v': dm.phase.options.StateOptionsDictionary()}

        state_options['x']['shape'] = (1,)
        state_options['x']['units'] = 'm'
        state_options['x']['rate_source'] = 'xdot'
        state_options['x']['targets'] = []

        state_options['y']['shape'] = (1,)
        state_options['y']['units'] = 'm'
        state_options['y']['rate_source'] = 'ydot'
        state_options['y']['targets'] = []

        state_options['v']['shape'] = (1,)
        state_options['v']['units'] = 'm/s'
        state_options['v']['rate_source'] = 'vdot'
        state_options['v']['targets'] = ['v']

        param_options = {'g': dm.phase.options.ParameterOptionsDictionary()}

        param_options['g']['shape'] = (1,)
        param_options['g']['units'] = 'm/s**2'
        param_options['g']['targets'] = ['g']

        control_options = {'theta': dm.phase.options.ControlOptionsDictionary()}

        control_options['theta']['shape'] = (1,)
        control_options['theta']['units'] = 'rad'
        control_options['theta']['targets'] = ['theta']

        polynomial_control_options = {}

        p = om.Problem()

        p.model.add_subsystem('fixed_step_integrator', EulerIntegrationComp(ode_class=BrachistochroneODE,
                                                                            time_options=time_options,
                                                                            state_options=state_options,
                                                                            parameter_options=param_options,
                                                                            control_options=control_options,
                                                                            polynomial_control_options=polynomial_control_options,
                                                                            mode='fwd',
                                                                            num_steps_per_segment=10,
                                                                            grid_data=gd,
                                                                            ode_init_kwargs=None,
                                                                            complex_step_mode=True))
        p.setup(mode='fwd', force_alloc_complex=True)

        p.set_val('fixed_step_integrator.state_initial_values:x', 0.0)
        p.set_val('fixed_step_integrator.state_initial_values:y', 10.0)
        p.set_val('fixed_step_integrator.state_initial_values:v', 0.0)
        p.set_val('fixed_step_integrator.t_initial', 0.0)
        p.set_val('fixed_step_integrator.t_duration', 1.8016)
        p.set_val('fixed_step_integrator.parameters:g', 9.80665)
        p.set_val('fixed_step_integrator.controls:theta', np.linspace(1.0, 100.0, 30), units='deg')

        p.run_model()

        x_f = p.get_val('fixed_step_integrator.state_final_values:x')
        y_f = p.get_val('fixed_step_integrator.state_final_values:y')
        v_f = p.get_val('fixed_step_integrator.state_final_values:v')

        # These tolerances are loose since theta is not properly spaced along the lgl nodes.
        assert_near_equal(x_f, 10.0, tolerance=0.1)
        assert_near_equal(y_f, 5.0, tolerance=0.1)
        assert_near_equal(v_f, 9.9, tolerance=0.1)

        with np.printoptions(linewidth=1024):
            p.check_partials(compact_print=False, method='cs')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
