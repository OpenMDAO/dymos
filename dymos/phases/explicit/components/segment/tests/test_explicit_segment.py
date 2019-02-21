from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos import declare_state, declare_time
from dymos.phases.options import TimeOptionsDictionary
from dymos.phases.explicit.components.segment.explicit_segment import ExplicitSegment
from dymos.phases.explicit.solvers.nl_rk_solver import NonlinearRK
from dymos.phases.grid_data import GridData
from dymos.phases.simulation.ode_integration_interface import ODEIntegrationInterface


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


class TestExplicitSegmentSimpleIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        time_opts = TimeOptionsDictionary()
        time_opts.update(TestODE.ode_options._time_options)
        state_opts = TestODE.ode_options._states

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('seg_t0_tf', val=np.array([0.0, 2.0]), units='s')
        ivc.add_output('y_0', val=0.5, units='m')

        gd = GridData(num_segments=1, transcription='explicit', transcription_order=4)

        seg = ExplicitSegment(index=0, num_steps=4, method='rk4', state_options=state_opts,
                              time_options=time_opts, ode_class=TestODE, grid_data=gd)

        # Connect the state rates since this is usually handled at the phase level
        src_idxs = np.arange(4 * 4, dtype=int).reshape((4, 4, 1))
        seg.connect('stage_ode.ydot', 'state_rates:y', src_indices=src_idxs, flat_src_indices=True)

        cls.p.model.add_subsystem('segment', seg)

        cls.p.model.connect('seg_t0_tf', 'segment.seg_t0_tf')
        cls.p.model.connect('y_0', 'segment.initial_states:y')

    def test_results_nlbgs(self):
        """make sure you get the right answer using the NLBGS solver"""
        self.p.setup(check=True, force_alloc_complex=True)
        self.p.model.segment.nonlinear_solver.options['iprint'] = 2

        self.p.run_model()

        y_expected = np.array([[0.5],
                               [1.425130208333333],
                               [2.639602661132812],
                               [4.006818970044454],
                               [5.301605229265987]])

        assert_rel_error(self,
                         self.p['segment.step_states:y'],
                         y_expected,
                         tolerance=1.0E-9)

    def test_results_nlrk(self):
        """make sure you get the right answer using the custom nl_rk solver"""

        self.p.setup(check=False, force_alloc_complex=True)

        seg = self.p.model.segment

        seg.nonlinear_solver = NonlinearRK()

        self.p.run_model()

        y_expected = np.array([[0.5],
                               [1.425130208333333],
                               [2.639602661132812],
                               [4.006818970044454],
                               [5.301605229265987]])

        assert_rel_error(self,
                         self.p['segment.step_states:y'],
                         y_expected,
                         tolerance=1.0E-9)

    def test_partials(self):

        self.p.setup(check=False, force_alloc_complex=True)
        self.p.model.segment.nonlinear_solver.options['iprint'] = -1

        self.p.run_model()

        cpd = self.p.check_partials(method='cs', compact_print=True, out_stream=None)
        assert_check_partials(cpd)


if __name__ == "__main__":
    unittest.main()
