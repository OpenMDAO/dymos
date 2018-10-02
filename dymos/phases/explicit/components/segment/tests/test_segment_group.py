from __future__ import print_function, division,absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp, NonlinearBlockGS, NewtonSolver

from dymos import declare_state, declare_time
from dymos.phases.options import TimeOptionsDictionary
from dymos.phases.explicit.components.segment.segment_group import ExplicitSegment


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


class TestExplicitSegment(unittest.TestCase):

    def test_simple_integration(self):

        time_opts = TimeOptionsDictionary()
        time_opts.update(TestODE.ode_options._time_options)
        state_opts = TestODE.ode_options._states

        p = Problem(model=Group())

        print(repr(time_opts))
        print(state_opts)

        seg = ExplicitSegment(num_steps=4, method='rk4', state_options=state_opts,
                              time_options=time_opts, ode_class=TestODE)

        p.model.add_subsystem('segment', seg)

        p.setup(check=True)

        p.run_model()




