from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos import declare_time, declare_state
from dymos.phases.runge_kutta.segment.runge_kutta_k_comp import RungeKuttaKComp


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


class TestRKKComp(unittest.TestCase):

    def test_rk_k_comp_rk4_scalar(self):
        state_options = {'y': {'shape': (1,), 'units': 'm'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('h', shape=(1,), units='s')
        ivc.add_output('f:y', shape=(4, 1), units='m')

        p.model.add_subsystem('c', RungeKuttaKComp(method='rk4', state_options=state_options))

        p.model.connect('f:y', 'c.f:y')
        p.model.connect('h', 'c.h')

        p.setup(check=True, force_alloc_complex=True)

        p['h'] = 0.5
        p['f:y'] = 2.0*np.array([[0.75, 0.90625, 0.9453125, 1.09765625]]).T

        p.run_model()

        assert_rel_error(self,
                         p.get_val('c.k:y'),
                         np.array([[0.75, 0.90625, 0.9453125, 1.09765625]]).T)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_advance_comp_rk4_vector(self):
        state_options = {'y': {'shape': (2,), 'units': 'm'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('h', shape=(1,), units='s')
        ivc.add_output('f:y', shape=(4, 2), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaKComp(method='rk4', state_options=state_options))

        p.model.connect('f:y', 'c.f:y')
        p.model.connect('h', 'c.h')

        p.setup(check=True, force_alloc_complex=True)

        p['h'] = 0.5
        p['f:y'] = 2.0*np.array([[0.75, 0.90625, 0.9453125, 1.09765625],
                                 [1.087565104166667, 1.203206380208333, 1.23211669921875,
                                  1.328623453776042]]).T

        p.run_model()

        assert_rel_error(self,
                         p.get_val('c.k:y'),
                         np.array([[0.75, 0.90625, 0.9453125, 1.09765625],
                                   [1.087565104166667, 1.203206380208333, 1.23211669921875,
                                    1.328623453776042]]).T)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_advance_comp_rk4_matrix(self):
        state_options = {'y': {'shape': (2, 2), 'units': 'm'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('f:y', shape=(4, 2, 2), units='m')
        ivc.add_output('h', shape=(1,), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaKComp(method='rk4', state_options=state_options))

        p.model.connect('f:y', 'c.f:y')
        p.model.connect('h', 'c.h')

        p.setup(check=True, force_alloc_complex=True)

        p['h'] = 0.5
        p['f:y'] = 2.0*np.array([
                                [[0.75, 1.087565104166667],
                                 [1.319801330566406, 1.378409485022227]],

                                [[0.90625, 1.203206380208333],
                                 [1.368501663208008,  1.316761856277783]],

                                [[0.9453125, 1.23211669921875],
                                 [1.380676746368408,  1.301349949091673]],

                                [[1.09765625, 1.328623453776042],
                                 [1.385139703750610,  1.154084459568063]]])

        p.run_model()

        p.model.list_outputs(print_arrays=True)

        assert_rel_error(self,
                         p.get_val('c.k:y'),
                         np.array([
                             [[0.75, 1.087565104166667],
                              [1.319801330566406, 1.378409485022227]],

                             [[0.90625, 1.203206380208333],
                              [1.368501663208008, 1.316761856277783]],

                             [[0.9453125, 1.23211669921875],
                              [1.380676746368408, 1.301349949091673]],

                             [[1.09765625, 1.328623453776042],
                              [1.385139703750610, 1.154084459568063]]]))

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
