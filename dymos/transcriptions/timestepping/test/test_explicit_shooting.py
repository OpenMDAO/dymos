import unittest

import numpy as np

import openmdao.api as om
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestExplicitShooting(unittest.TestCase):

    def test_brachistochrone_explicit_shooting(self):
        prob = om.Problem()

        tx = dm.transcriptions.ExplicitShooting(num_segments=10, grid='gauss-lobatto',
                                                order=3, num_steps_per_segment=10)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        phase.set_time_options(units='s')

        # automatically discover states

        phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)
        phase.add_control('theta', val=45.0, units='deg', opt=True)

        prob.model.add_subsystem('phase0', phase)

        prob.setup()




        # state_options = {'x': dm.phase.options.StateOptionsDictionary(),
        #                  'y': dm.phase.options.StateOptionsDictionary(),
        #                  'v': dm.phase.options.StateOptionsDictionary()}
        #
        # state_options['x']['shape'] = (1,)
        # state_options['x']['units'] = 'm'
        # state_options['x']['rate_source'] = 'xdot'
        # state_options['x']['targets'] = []
        #
        # state_options['y']['shape'] = (1,)
        # state_options['y']['units'] = 'm'
        # state_options['y']['rate_source'] = 'ydot'
        # state_options['y']['targets'] = []
        #
        # state_options['v']['shape'] = (1,)
        # state_options['v']['units'] = 'm/s'
        # state_options['v']['rate_source'] = 'vdot'
        # state_options['v']['targets'] = ['v']
        #
        # param_options = {'g': dm.phase.options.ParameterOptionsDictionary()}
        #
        # # param_options['g']['shape'] = (1,)
        # # param_options['g']['units'] = 'm/s**2'
        # # param_options['g']['targets'] = ['g']
        #
        # control_options = {'theta': dm.phase.options.ControlOptionsDictionary()}
        #
        # control_options['theta']['shape'] = (1,)
        # control_options['theta']['units'] = 'rad'
        # control_options['theta']['targets'] = ['theta']
        #
        # polynomial_control_options = {}

        # p = om.Problem()
        #
        # p.model.add_subsystem('fixed_step_integrator', EulerIntegrationComp(ode_class=BrachistochroneODE,
        #                                                                     time_options=time_options,
        #                                                                     state_options=state_options,
        #                                                                     parameter_options=param_options,
        #                                                                     control_options=control_options,
        #                                                                     polynomial_control_options=polynomial_control_options,
        #                                                                     mode='fwd', num_steps=10,
        #                                                                     grid_data=gd,
        #                                                                     ode_init_kwargs=None,
        #                                                                     complex_step_mode=True))
        # p.setup(mode='fwd', force_alloc_complex=True)
        #
        # p.set_val('fixed_step_integrator.state_initial_values:x', 0.0)
        # p.set_val('fixed_step_integrator.state_initial_values:y', 10.0)
        # p.set_val('fixed_step_integrator.state_initial_values:v', 0.0)
        # p.set_val('fixed_step_integrator.t_initial', 0.0)
        # p.set_val('fixed_step_integrator.t_duration', 1.8016)
        # p.set_val('fixed_step_integrator.parameters:g', 9.80665)
        # p.set_val('fixed_step_integrator.controls:theta', np.linspace(1.0, 100.0, 30), units='deg')
        #
        # p.run_model()
        #
        # x_f = p.get_val('fixed_step_integrator.state_final_values:x')
        # y_f = p.get_val('fixed_step_integrator.state_final_values:y')
        # v_f = p.get_val('fixed_step_integrator.state_final_values:v')
        #
        # # These tolerances are loose since theta is not properly spaced along the lgl nodes.
        # assert_near_equal(x_f, 10.0, tolerance=0.1)
        # assert_near_equal(y_f, 5.0, tolerance=0.1)
        # assert_near_equal(v_f, 9.9, tolerance=0.1)
        #
        # with np.printoptions(linewidth=1024):
        #     p.check_partials(compact_print=False, method='cs')
