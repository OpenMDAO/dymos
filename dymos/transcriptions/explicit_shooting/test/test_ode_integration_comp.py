import unittest

import numpy as np
import openmdao.api as om
import dymos as dm
from dymos import options as dymos_options

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.transcriptions.explicit_shooting.ode_integration_comp import ODEIntegrationComp

from dymos.phase.options import TimeOptionsDictionary, StateOptionsDictionary, ParameterOptionsDictionary
from dymos.transcriptions.grid_data import GridData


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


class TestODEIntegrationComp(unittest.TestCase):

    def test_integrate_scalar_ode(self):
        dymos_options['include_check_partials'] = True

        for ogd_eq_igd in (True, False):
            with self.subTest('input_grid_data = output_grid_data'
                              if ogd_eq_igd else 'input_grid_data != output_grid_data'):

                input_grid_data = GridData(num_segments=1, transcription='gauss-lobatto', transcription_order=3)

                if ogd_eq_igd:
                    output_grid_data = input_grid_data
                else:
                    output_grid_data = GridData(num_segments=1, transcription='uniform', transcription_order=10)

                time_options = TimeOptionsDictionary()

                time_options['targets'] = 't'
                time_options['units'] = 's'

                state_options = {'x': StateOptionsDictionary()}

                state_options['x']['shape'] = (1,)
                state_options['x']['units'] = 's**2'
                state_options['x']['rate_source'] = 'x_dot'
                state_options['x']['targets'] = ['x']

                param_options = {'p': ParameterOptionsDictionary()}

                param_options['p']['shape'] = (1,)
                param_options['p']['units'] = 's**2'
                param_options['p']['targets'] = ['p']

                control_options = {}
                polynomial_control_options = {}

                prob = om.Problem()

                prob.model.add_subsystem('integrator',
                                         ODEIntegrationComp(input_grid_data=input_grid_data,
                                                            output_grid_data=output_grid_data,
                                                            time_options=time_options, state_options=state_options,
                                                            parameter_options=param_options,
                                                            control_options=control_options,
                                                            polynomial_control_options=polynomial_control_options,
                                                            ode_class=SimpleODE, ode_init_kwargs=None))
                prob.setup()
                prob.set_val('integrator.states:x', 0.5)
                prob.set_val('integrator.t_initial', 0.0)
                prob.set_val('integrator.t_duration', 2.0)
                prob.set_val('integrator.parameters:p', 1.0)

                prob.run_model()

                x = prob.get_val('integrator.states_out:x')
                t = prob.get_val('integrator.time')
                p = prob.get_val('integrator.parameters:p')

                expected = t**2 + 2 * t + p - 0.5 * np.exp(t)

                assert_near_equal(x, expected, tolerance=1.0E-5)

                cpd = prob.check_partials(compact_print=True)
                assert_check_partials(cpd, atol=1.0E-4, rtol=1.0E-5)

        dymos_options['include_check_partials'] = False

    def test_integrate_with_controls(self):

        dymos_options['include_check_partials'] = True

        gd = dm.transcriptions.grid_data.GridData(num_segments=5, transcription='gauss-lobatto',
                                                  transcription_order=3, compressed=True)

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

        p.model.add_subsystem('integrator',
                              ODEIntegrationComp(ode_class=BrachistochroneODE,
                                                 time_options=time_options,
                                                 state_options=state_options,
                                                 parameter_options=param_options,
                                                 control_options=control_options,
                                                 polynomial_control_options=polynomial_control_options,
                                                 input_grid_data=gd,
                                                 ode_init_kwargs=None))

        p.setup()

        p.set_val('integrator.states:x', 0.0)
        p.set_val('integrator.states:y', 10.0)
        p.set_val('integrator.states:v', 0.0)
        p.set_val('integrator.t_initial', 0.0)
        p.set_val('integrator.t_duration', 1.8016)
        p.set_val('integrator.parameters:g', 9.80665)

        p.set_val('integrator.controls:theta', np.linspace(0.01, 100.0, gd.subset_num_nodes['control_input']),
                  units='deg')

        p.run_model()

        t = p.get_val('integrator.time')
        x = p.get_val('integrator.states_out:x')
        y = p.get_val('integrator.states_out:y')
        v = p.get_val('integrator.states_out:v')

        # These tolerances are loose since theta is not properly spaced along the lgl nodes.
        assert_near_equal(x[-1, ...], 10.0, tolerance=0.1)
        assert_near_equal(y[-1, ...], 5.0, tolerance=0.1)
        assert_near_equal(v[-1, ...], 9.9, tolerance=0.1)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='fd')
            assert_check_partials(cpd, atol=1.0E-4, rtol=1.0E-4)

    def test_integrate_with_polynomial_controls(self):

        dymos_options['include_check_partials'] = True

        gd = dm.transcriptions.grid_data.GridData(num_segments=5, transcription='gauss-lobatto',
                                                  transcription_order=3, compressed=True)

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

        poly_control_options = {'theta': dm.phase.options.PolynomialControlOptionsDictionary()}

        poly_control_options['theta']['shape'] = (1,)
        poly_control_options['theta']['order'] = 2
        poly_control_options['theta']['units'] = 'rad'
        poly_control_options['theta']['targets'] = ['theta']

        control_options = {}

        p = om.Problem()

        p.model.add_subsystem('integrator',
                              ODEIntegrationComp(ode_class=BrachistochroneODE,
                                                 time_options=time_options,
                                                 state_options=state_options,
                                                 parameter_options=param_options,
                                                 control_options=control_options,
                                                 polynomial_control_options=poly_control_options,
                                                 input_grid_data=gd,
                                                 ode_init_kwargs=None))

        p.setup()

        p.set_val('integrator.states:x', 0.0)
        p.set_val('integrator.states:y', 10.0)
        p.set_val('integrator.states:v', 0.0)
        p.set_val('integrator.t_initial', 0.0)
        p.set_val('integrator.t_duration', 1.8016)
        p.set_val('integrator.parameters:g', 9.80665)

        p.set_val('integrator.polynomial_controls:theta',
                  np.linspace(0.01, 100.0, poly_control_options['theta']['order']+1),
                  units='deg')

        p.run_model()

        t = p.get_val('integrator.time')
        x = p.get_val('integrator.states_out:x')
        y = p.get_val('integrator.states_out:y')
        v = p.get_val('integrator.states_out:v')

        # These tolerances are loose since theta is not properly spaced along the lgl nodes.
        assert_near_equal(x[-1, ...], 10.0, tolerance=0.1)
        assert_near_equal(y[-1, ...], 5.0, tolerance=0.1)
        assert_near_equal(v[-1, ...], 9.9, tolerance=0.1)

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(compact_print=False, method='fd')
            assert_check_partials(cpd, atol=1.0E-4, rtol=1.0E-4)

        dymos_options['include_check_partials'] = False


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
