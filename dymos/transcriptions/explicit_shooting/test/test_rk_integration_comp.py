import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.transcriptions.explicit_shooting.rk_integration_comp import RKIntegrationComp

from openmdao.utils.testing_utils import require_pyoptsparse


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


class TestRKIntegrationComp(unittest.TestCase):

    # @require_pyoptsparse(optimizer='SNOPT')
    def test_eval_f_scalar(self):
        gd = dm.transcriptions.grid_data.GridData(num_segments=10, transcription='gauss-lobatto',
                                                  transcription_order=3)

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
                                 RKIntegrationComp(SimpleODE, time_options, state_options,
                                                   param_options, control_options,
                                                   polynomial_control_options,
                                                   grid_data=gd,
                                                   num_steps_per_segment=100))
        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.set_val('fixed_step_integrator.states:x', 0.5)
        prob.set_val('fixed_step_integrator.t_initial', 0.0)
        prob.set_val('fixed_step_integrator.t_duration', 2.0)
        prob.set_val('fixed_step_integrator.parameters:p', 1.0)

        x = np.array([[0.5]])
        t = np.array([[0.0]])
        theta = np.array([[0.0, 2.0, 1.0]]).T
        f = np.zeros_like(x)

        intg = prob.model.fixed_step_integrator
        intg._eval_subprob.model.ode_eval.set_segment_index(9)
        intg.eval_f(x, t, theta, f)

        assert_near_equal(f, x - t**2 + theta[2, 0])

    # @require_pyoptsparse(optimizer='SNOPT')
    def test_eval_f_derivs_scalar(self):
        gd = dm.transcriptions.grid_data.GridData(num_segments=10, transcription='gauss-lobatto',
                                                  transcription_order=3)

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
                                 RKIntegrationComp(SimpleODE, time_options, state_options,
                                                   param_options, control_options,
                                                   polynomial_control_options,
                                                   grid_data=gd,
                                                   num_steps_per_segment=100))
        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.set_val('fixed_step_integrator.states:x', 0.5)
        prob.set_val('fixed_step_integrator.t_initial', 0.0)
        prob.set_val('fixed_step_integrator.t_duration', 2.0)
        prob.set_val('fixed_step_integrator.parameters:p', 1.0)

        prob.final_setup()

        prob.run_model()

        x = np.array([[0.5]])
        t = np.array([[0.0]])
        theta = np.array([[0, 2.0, 1.0]]).T

        f_nom = np.zeros_like(x)

        f_x_fd = np.zeros_like(x)
        f_t_fd = np.zeros_like(x)
        f_theta_fd = np.zeros_like(x)

        intg = prob.model.fixed_step_integrator
        f_x = intg._f_x
        f_t = intg._f_t
        f_theta = intg._f_θ

        intg._eval_subprob.model.ode_eval.set_segment_index(9)

        intg.eval_f(x, t, theta, f_nom)
        intg.eval_f_derivs(x, t, theta, f_x, f_t, f_theta)

        step = 1.0E-8

        intg.eval_f(x + step, t, theta, f_x_fd)
        f_x_fd = (f_x_fd - f_nom) / step
        assert_near_equal(f_x, f_x_fd, tolerance=1.0E-6)

        intg.eval_f(x, t + step, theta, f_t_fd)
        f_t_fd = (f_t_fd - f_nom) / step
        assert_near_equal(f_t, f_t_fd, tolerance=1.0E-6)

        theta[2] = theta[2] + step
        prob.model.fixed_step_integrator.eval_f(x, t, theta, f_theta_fd)
        f_theta_fd = (f_theta_fd - f_nom) / step
        assert_near_equal(f_theta.real[0, 2], f_theta_fd[0, 0], tolerance=1.0E-6)

    # @require_pyoptsparse(optimizer='SNOPT')
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

        gd = dm.transcriptions.grid_data.GridData(num_segments=10,
                                                  transcription='gauss-lobatto',
                                                  transcription_order=3,
                                                  compressed=True)

        p.model.add_subsystem('fixed_step_integrator', RKIntegrationComp(SimpleODE, time_options, state_options,
                                                                         param_options, control_options,
                                                                         polynomial_control_options,
                                                                         grid_data=gd,
                                                                         num_steps_per_segment=10,
                                                                         ode_init_kwargs=None))
        p.setup(mode='fwd', force_alloc_complex=True)

        p.set_val('fixed_step_integrator.states:x', 0.5)
        p.set_val('fixed_step_integrator.t_initial', 0.0)
        p.set_val('fixed_step_integrator.t_duration', 2.0)
        p.set_val('fixed_step_integrator.parameters:p', 1.0)

        p.run_model()

        t = p.get_val('fixed_step_integrator.time')
        x = p.get_val('fixed_step_integrator.states_out:x')

        assert_near_equal(t[-1, ...], 2)
        assert_near_equal(x[-1, ...], 5.305471950534675, tolerance=1.0E-7)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(method='cs', compact_print=True)
            assert_check_partials(cpd)

    # @require_pyoptsparse(optimizer='SNOPT')
    def test_fwd_parameters_controls(self):
        gd = dm.transcriptions.grid_data.GridData(num_segments=5, transcription='gauss-lobatto',
                                                  transcription_order=5, compressed=True)

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

        p.model.add_subsystem('fixed_step_integrator',
                              RKIntegrationComp(ode_class=BrachistochroneODE,
                                                time_options=time_options,
                                                state_options=state_options,
                                                parameter_options=param_options,
                                                control_options=control_options,
                                                polynomial_control_options=polynomial_control_options,
                                                num_steps_per_segment=10,
                                                grid_data=gd,
                                                ode_init_kwargs=None))

        p.setup(mode='fwd', force_alloc_complex=True)

        p.set_val('fixed_step_integrator.states:x', 0.0)
        p.set_val('fixed_step_integrator.states:y', 10.0)
        p.set_val('fixed_step_integrator.states:v', 0.0)
        p.set_val('fixed_step_integrator.t_initial', 0.0)
        p.set_val('fixed_step_integrator.t_duration', 1.8016)
        p.set_val('fixed_step_integrator.parameters:g', 9.80665)

        p.set_val('fixed_step_integrator.controls:theta', np.linspace(0.01, 100.0, 21), units='deg')

        p.run_model()

        x = p.get_val('fixed_step_integrator.states_out:x')
        y = p.get_val('fixed_step_integrator.states_out:y')
        v = p.get_val('fixed_step_integrator.states_out:v')
        theta = p.get_val('fixed_step_integrator.control_values:theta')

        # These tolerances are loose since theta is not properly spaced along the lgl nodes.
        assert_near_equal(x[-1, ...], 10.0, tolerance=0.1)
        assert_near_equal(y[-1, ...], 5.0, tolerance=0.1)
        assert_near_equal(v[-1, ...], 9.9, tolerance=0.1)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=True, method='cs', show_only_incorrect=True)
            assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
