from __future__ import division, print_function, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.options import TimeOptionsDictionary, StateOptionsDictionary, \
    ControlOptionsDictionary
from dymos.phases.components import EndpointConditionsComp


class TestEndpointConditionComp(unittest.TestCase):

    def test_scalar_state_and_control(self):
        n = 101

        p = Problem(model=Group())

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

        ivc = IndepVarComp()
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

        p.model.add_subsystem('end_conditions',
                              subsys=EndpointConditionsComp(time_options=time_options,
                                                            state_options=state_options,
                                                            control_options=control_options))

        p.model.connect('phase:time', 'end_conditions.values:time')

        size = np.prod(state_options['x']['shape'])

        p.model.connect('phase:x', 'end_conditions.values:x')

        p.model.connect('phase:theta', 'end_conditions.values:theta')

        p.model.connect('phase:initial_jump:time', 'end_conditions.initial_jump:time')
        p.model.connect('phase:final_jump:time', 'end_conditions.final_jump:time')

        p.model.connect('phase:initial_jump:x', 'end_conditions.initial_jump:x')
        p.model.connect('phase:final_jump:x', 'end_conditions.final_jump:x')

        p.model.connect('phase:initial_jump:theta', 'end_conditions.initial_jump:theta')
        p.model.connect('phase:final_jump:theta', 'end_conditions.final_jump:theta')

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'dense'

        p.setup(mode='fwd', force_alloc_complex=True)

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

        assert_almost_equal(p['end_conditions.time--'],
                            p['phase:time'][0] - p['phase:initial_jump:time'])

        assert_almost_equal(p['end_conditions.time-+'],
                            p['phase:time'][0])

        assert_almost_equal(p['end_conditions.time++'],
                            p['phase:time'][-1] + p['phase:final_jump:time'])

        assert_almost_equal(p['end_conditions.time+-'],
                            p['phase:time'][-1])

        assert_almost_equal(p['end_conditions.states:x--'],
                            p['phase:x'][0] - p['phase:initial_jump:x'])

        assert_almost_equal(p['end_conditions.states:x++'],
                            p['phase:x'][-1] + p['phase:final_jump:x'])

        assert_almost_equal(p['end_conditions.controls:theta--'],
                            p['phase:theta'][0] - p['phase:initial_jump:theta'])

        assert_almost_equal(p['end_conditions.controls:theta++'],
                            p['phase:theta'][-1] + p['phase:final_jump:theta'])

        cpd = p.check_partials(compact_print=True, method='cs')

        assert_check_partials(cpd)

    def test_vector_state_and_control(self):
        n = 101

        p = Problem(model=Group())

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

        ivc = IndepVarComp()
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

        p.model.add_subsystem('end_conditions',
                              subsys=EndpointConditionsComp(time_options=time_options,
                                                            state_options=state_options,
                                                            control_options=control_options))

        p.model.connect('phase:time', 'end_conditions.values:time')

        p.model.connect('phase:pos', 'end_conditions.values:pos')

        p.model.connect('phase:initial_jump:pos', 'end_conditions.initial_jump:pos')
        p.model.connect('phase:final_jump:pos', 'end_conditions.final_jump:pos')

        p.model.connect('phase:cmd', 'end_conditions.values:cmd')

        p.model.connect('phase:initial_jump:cmd', 'end_conditions.initial_jump:cmd')
        p.model.connect('phase:final_jump:cmd', 'end_conditions.final_jump:cmd')

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(mode='fwd', force_alloc_complex=True)

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

        assert_almost_equal(p['end_conditions.states:pos--'],
                            p['phase:pos'][0, :] - p['phase:initial_jump:pos'])

        assert_almost_equal(p['end_conditions.states:pos++'],
                            p['phase:pos'][-1, :] + p['phase:final_jump:pos'])

        assert_almost_equal(p['end_conditions.controls:cmd--'],
                            p['phase:cmd'][0, :] - p['phase:initial_jump:cmd'])

        assert_almost_equal(p['end_conditions.controls:cmd++'],
                            p['phase:cmd'][-1, :] + p['phase:final_jump:cmd'])

        cpd = p.check_partials(compact_print=True, method='cs')

        assert_check_partials(cpd)

if __name__ == "__main__":
    unittest.main()
