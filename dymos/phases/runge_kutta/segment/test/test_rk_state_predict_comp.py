from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from dymos import declare_time, declare_state
from dymos.phases.runge_kutta.segment.runge_kutta_state_predict_comp import \
    RungeKuttaStatePredictComp


@declare_time(targets=['t'], units='s')
@declare_state('y', targets=['y'], rate_source='ydot', units='m')
class TestODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        self.add_input('t', val=np.ones(self.options['num_nodes']), units='s')
        self.add_input('y', val=np.ones(self.options['num_nodes']), units='m')
        self.add_output('ydot', val=np.ones(self.options['num_nodes']), units='m/s')

        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='ydot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='ydot', wrt='y', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        t = inputs['t']
        y = inputs['y']
        outputs['ydot'] = y - t ** 2 + 1

    def compute_partials(self, inputs, partials):
        partials['ydot', 't'] = -2 * inputs['t']


class TestRKStatePredictComp(unittest.TestCase):

    def test_rk_state_predict_comp_rk4(self):
        state_options = {'y': {'shape': (1,), 'units': 'm'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(4, 1), units='m')
        ivc.add_output('y0', shape=(1,), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStatePredictComp(method='rk4', state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states:y')

        p.setup(check=True)

        p['y0'] = 0.5
        p['k:y'] = np.array([[0.75, 0.90625, 0.9453125, 1.09765625]]).T

        p.run_model()

        assert_rel_error(self,
                         p.get_val('c.predicted_states:y'),
                         np.array([[0.5, 0.875, 0.953125, 1.4453125]]).T)
