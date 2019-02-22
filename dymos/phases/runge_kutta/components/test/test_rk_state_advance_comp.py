from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.phases.runge_kutta.components.runge_kutta_state_advance_comp import \
    RungeKuttaStateAdvanceComp


class TestRKStateAdvanceComp(unittest.TestCase):

    def test_rk_state_advance_comp_rk4_scalar(self):
        state_options = {'y': {'shape': (1,), 'units': 'm'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(4, 4, 1), units='m')
        ivc.add_output('y0', shape=(4, 1), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStateAdvanceComp(num_segments=4, method='rk4',
                                                         state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states:y')

        p.setup(check=True, force_alloc_complex=True)

        p['y0'] = np.array([[0.5, 1.425130208333333, 2.639602661132812, 4.006818970044454]]).T
        p['k:y'] = np.array([[[0.75000000],
                              [0.90625000],
                              [0.94531250],
                              [1.09765625]],

                             [[1.08756510],
                              [1.20320638],
                              [1.23211670],
                              [1.32862345]],

                             [[1.31980133],
                              [1.36850166],
                              [1.38067675],
                              [1.38513970]],

                             [[1.37840949],
                              [1.31676186],
                              [1.30134995],
                              [1.15408446]]])

        p.run_model()

        assert_rel_error(self,
                         p.get_val('c.final_states:y'),
                         np.array([[1.425130208333333],
                                   [2.639602661132812],
                                   [4.006818970044454],
                                   [5.301605229265987]]),
                         tolerance=1.0E-9)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)

    def test_rk_state_advance_comp_rk4_vector(self):
        state_options = {'y': {'shape': (2,), 'units': 'm'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(2, 4, 2), units='m')
        ivc.add_output('y0', shape=(2, 2), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStateAdvanceComp(num_segments=2, method='rk4',
                                                         state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states:y')

        p.setup(check=True, force_alloc_complex=True)

        p['y0'] = [[0.5, 1.425130208333333], [2.639602661132812, 4.006818970044454]]
        p['k:y'][0, :, 0] = [0.75, 0.90625, 0.9453125, 1.09765625]
        p['k:y'][0, :, 1] = [1.08756510, 1.20320638, 1.23211670, 1.32862345]
        p['k:y'][1, :, 0] = [1.31980133, 1.36850166, 1.38067675, 1.38513970]
        p['k:y'][1, :, 1] = [1.37840949, 1.31676186, 1.30134995, 1.15408446]

        p.run_model()

        p.model.list_outputs(print_arrays=True)

        assert_rel_error(test_case=self,
                         actual=p.get_val('c.final_states:y'),
                         desired=[[1.425130208333333, 2.639602661132812],
                                  [4.006818970044454, 5.301605229265987]],
                         tolerance=1.0E-9)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_advance_comp_rk4_matrix(self):
        state_options = {'y': {'shape': (2, 2), 'units': 'm'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(4, 2, 2), units='m')
        ivc.add_output('y0', shape=(2, 2), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStateAdvanceComp(num_segments=1,
                                                         method='rk4',
                                                         state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states:y')

        p.setup(check=True, force_alloc_complex=True)

        p['y0'] = [[0.5, 1.425130208333333], [2.639602661132812, 4.006818970044454]]
        p['k:y'] = np.array([
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
                         p.get_val('c.final_states:y'),
                         [[[1.425130208333333, 2.639602661132812],
                          [4.006818970044454, 5.301605229265987]]])

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
