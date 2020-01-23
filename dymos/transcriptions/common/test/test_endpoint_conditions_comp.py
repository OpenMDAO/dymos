import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phase.options import TimeOptionsDictionary, StateOptionsDictionary, \
    ControlOptionsDictionary
from dymos.transcriptions.common import EndpointConditionsComp


class TestEndpointConditionComp(unittest.TestCase):

    def test_scalar_state_and_control(self):
        n = 101

        p = om.Problem(model=om.Group())

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        state_options = {}
        state_options['x'] = StateOptionsDictionary()
        state_options['x']['units'] = 'm'
        state_options['x']['shape'] = (1,)

        control_options = {}
        control_options['theta'] = ControlOptionsDictionary()
        control_options['theta']['units'] = 'rad'
        control_options['theta']['shape'] = (1,)

        ivc = om.IndepVarComp()
        ivc.add_output(name='phase:time', val=np.zeros(n), units='s')
        ivc.add_output(name='phase:initial_jump:time', val=100.0 * np.ones(1), units='s')
        ivc.add_output(name='phase:final_jump:time', val=1.0 * np.ones(1), units='s')

        ivc.add_output(name='phase:x',
                       val=np.zeros(n),
                       units='m')

        ivc.add_output(name='phase:initial_jump:x',
                       val=np.zeros(state_options['x']['shape']),
                       units='m')

        ivc.add_output(name='phase:final_jump:x',
                       val=np.zeros(state_options['x']['shape']),
                       units='m')

        ivc.add_output(name='phase:theta',
                       val=np.zeros(n),
                       units='rad')

        ivc.add_output(name='phase:initial_jump:theta',
                       val=np.zeros(control_options['theta']['shape']),
                       units='rad')

        ivc.add_output(name='phase:final_jump:theta',
                       val=np.zeros(control_options['theta']['shape']),
                       units='rad')

        p.model.add_subsystem('ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem('initial_conditions',
                              subsys=EndpointConditionsComp(loc='initial',
                                                            time_options=time_options,
                                                            state_options=state_options,
                                                            control_options=control_options),
                              promotes_outputs=['*'])

        p.model.add_subsystem('final_conditions',
                              subsys=EndpointConditionsComp(loc='final',
                                                            time_options=time_options,
                                                            state_options=state_options,
                                                            control_options=control_options),
                              promotes_outputs=['*'])

        p.model.connect('phase:time', 'initial_conditions.initial_value:time')
        p.model.connect('phase:time', 'final_conditions.final_value:time')

        p.model.connect('phase:x', 'initial_conditions.initial_value:x')
        p.model.connect('phase:x', 'final_conditions.final_value:x')

        p.model.connect('phase:theta', 'initial_conditions.initial_value:theta')
        p.model.connect('phase:theta', 'final_conditions.final_value:theta')

        p.model.connect('phase:initial_jump:time', 'initial_conditions.initial_jump:time')
        p.model.connect('phase:final_jump:time', 'final_conditions.final_jump:time')

        p.model.connect('phase:initial_jump:x', 'initial_conditions.initial_jump:x')
        p.model.connect('phase:final_jump:x', 'final_conditions.final_jump:x')

        p.model.connect('phase:initial_jump:theta', 'initial_conditions.initial_jump:theta')
        p.model.connect('phase:final_jump:theta', 'final_conditions.final_jump:theta')

        p.model.linear_solver = om.DirectSolver()

        p.setup(force_alloc_complex=True)

        p['phase:time'] = np.linspace(0, 500, n)
        p['phase:time'] = np.linspace(0, 500, n)
        p['phase:x'] = np.linspace(0, 10, n)
        p['phase:theta'] = np.linspace(-1, 1, n)

        p['phase:initial_jump:time'] = 50.0
        p['phase:final_jump:time'] = 75.0

        p['phase:initial_jump:x'] = 100.0
        p['phase:final_jump:x'] = 200.0

        p['phase:initial_jump:theta'] = -1.0
        p['phase:final_jump:theta'] = -1.0

        p.run_model()

        assert_almost_equal(p['time--'],
                            p['phase:time'][0] - p['phase:initial_jump:time'])

        assert_almost_equal(p['time-+'],
                            p['phase:time'][0])

        assert_almost_equal(p['time++'],
                            p['phase:time'][-1] + p['phase:final_jump:time'])

        assert_almost_equal(p['time+-'],
                            p['phase:time'][-1])

        assert_almost_equal(p['states:x--'],
                            p['phase:x'][0] - p['phase:initial_jump:x'])

        assert_almost_equal(p['states:x++'],
                            p['phase:x'][-1] + p['phase:final_jump:x'])

        assert_almost_equal(p['controls:theta--'],
                            p['phase:theta'][0] - p['phase:initial_jump:theta'])

        assert_almost_equal(p['controls:theta++'],
                            p['phase:theta'][-1] + p['phase:final_jump:theta'])

        cpd = p.check_partials(compact_print=True, method='cs')

        assert_check_partials(cpd)

    def test_vector_state_and_control(self):
        n = 101

        p = om.Problem(model=om.Group())

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        state_options = {}
        state_options['pos'] = StateOptionsDictionary()
        state_options['pos']['units'] = 'm'
        state_options['pos']['shape'] = (3,)

        control_options = {}
        control_options['cmd'] = ControlOptionsDictionary()
        control_options['cmd']['units'] = 'rad'
        control_options['cmd']['shape'] = (3,)

        ivc = om.IndepVarComp()
        ivc.add_output(name='phase:time', val=np.zeros(n), units='s')
        ivc.add_output(name='phase:initial_jump:time', val=100.0 * np.ones(1), units='s')

        ivc.add_output(name='phase:pos',
                       val=np.zeros((n, 3)),
                       units='m')

        ivc.add_output(name='phase:initial_jump:pos',
                       val=np.zeros(state_options['pos']['shape']),
                       units='m')

        ivc.add_output(name='phase:final_jump:pos',
                       val=np.zeros(state_options['pos']['shape']),
                       units='m')

        ivc.add_output(name='phase:cmd',
                       val=np.zeros((n, 3)),
                       units='rad')

        ivc.add_output(name='phase:initial_jump:cmd',
                       val=np.zeros(control_options['cmd']['shape']),
                       units='rad')

        ivc.add_output(name='phase:final_jump:cmd',
                       val=np.zeros(control_options['cmd']['shape']),
                       units='rad')

        p.model.add_subsystem('ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem('initial_conditions',
                              subsys=EndpointConditionsComp(loc='initial',
                                                            time_options=time_options,
                                                            state_options=state_options,
                                                            control_options=control_options),
                              promotes_outputs=['*'])

        p.model.add_subsystem('final_conditions',
                              subsys=EndpointConditionsComp(loc='final',
                                                            time_options=time_options,
                                                            state_options=state_options,
                                                            control_options=control_options),
                              promotes_outputs=['*'])

        p.model.connect('phase:time', 'initial_conditions.initial_value:time')
        p.model.connect('phase:time', 'final_conditions.final_value:time')

        p.model.connect('phase:pos', 'initial_conditions.initial_value:pos')
        p.model.connect('phase:pos', 'final_conditions.final_value:pos')

        p.model.connect('phase:initial_jump:pos', 'initial_conditions.initial_jump:pos')
        p.model.connect('phase:final_jump:pos', 'final_conditions.final_jump:pos')

        p.model.connect('phase:cmd', 'initial_conditions.initial_value:cmd')
        p.model.connect('phase:cmd', 'final_conditions.final_value:cmd')

        p.model.connect('phase:initial_jump:cmd', 'initial_conditions.initial_jump:cmd')
        p.model.connect('phase:final_jump:cmd', 'final_conditions.final_jump:cmd')

        p.model.linear_solver = om.DirectSolver()
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(force_alloc_complex=True)

        p['phase:time'] = np.linspace(0, 500, n)
        p['phase:pos'][:, 0] = np.linspace(0, 10, n)
        p['phase:pos'][:, 1] = np.linspace(0, 20, n)
        p['phase:pos'][:, 2] = np.linspace(0, 30, n)

        p['phase:initial_jump:pos'] = np.array([7, 8, 9])
        p['phase:final_jump:pos'] = np.array([4, 5, 6])

        p['phase:cmd'][:, 0] = np.linspace(0, 5, n)
        p['phase:cmd'][:, 1] = np.linspace(0, 6, n)
        p['phase:cmd'][:, 2] = np.linspace(0, 7, n)

        p['phase:initial_jump:cmd'] = np.array([1, 2, 3])
        p['phase:final_jump:cmd'] = np.array([10, 11, 12])

        p.run_model()

        assert_almost_equal(p['states:pos--'],
                            p['phase:pos'][0, :] - p['phase:initial_jump:pos'])

        assert_almost_equal(p['states:pos++'],
                            p['phase:pos'][-1, :] + p['phase:final_jump:pos'])

        assert_almost_equal(p['controls:cmd--'],
                            p['phase:cmd'][0, :] - p['phase:initial_jump:cmd'])

        assert_almost_equal(p['controls:cmd++'],
                            p['phase:cmd'][-1, :] + p['phase:final_jump:cmd'])

        cpd = p.check_partials(compact_print=True, method='cs')

        assert_check_partials(cpd)


if __name__ == "__main__":
    unittest.main()
