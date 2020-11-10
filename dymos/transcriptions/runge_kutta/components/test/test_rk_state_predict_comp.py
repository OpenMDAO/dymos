import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.testing_utils import assert_check_partials

from dymos.transcriptions.runge_kutta.components.runge_kutta_state_predict_comp import \
    RungeKuttaStatePredictComp
import dymos as dm

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
RungeKuttaStatePredictComp = CompWrapperConfig(RungeKuttaStatePredictComp)


class TestRKStatePredictComp(unittest.TestCase):

    def setUp(self) -> None:
        dm.options['include_check_partials'] = True

    def tearDown(self) -> None:
        dm.options['include_check_partials'] = False

    def test_rk_state_predict_comp_rk4(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm'}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(num_seg, 4, 1), units='m')
        ivc.add_output('y0', shape=(num_seg, 1), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStatePredictComp(num_segments=4, method='RK4',
                                                         state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states_per_seg:y')

        p.setup(check=True, force_alloc_complex=True)

        p['y0'] = np.array([[0.5, 1.425130208333333, 2.639602661132812, 4.006818970044454]]).T
        p['k:y'] = np.array([[[0.75000000],
                              [0.90625000],
                              [0.94531250],
                              [1.09765625]],

                             [[1.087565104166667],
                              [1.203206380208333],
                              [1.232116699218750],
                              [1.328623453776042]],

                             [[1.319801330566406],
                              [1.368501663208008],
                              [1.380676746368408],
                              [1.385139703750610]],

                             [[1.378409485022227],
                              [1.316761856277783],
                              [1.301349949091673],
                              [1.154084459568063]]])

        p.run_model()

        p.model.list_outputs(print_arrays=True)

        expected = np.array([[0.50000000],
                             [0.87500000],
                             [0.953125],
                             [1.4453125],

                             [1.425130208333333],
                             [1.968912760416667],
                             [2.026733398437500],
                             [2.657246907552083],

                             [2.639602661132812],
                             [3.299503326416016],
                             [3.323853492736816],
                             [4.020279407501221],

                             [4.006818970044454],
                             [4.696023712555567],
                             [4.665199898183346],
                             [5.308168919136127]])

        assert_near_equal(p.get_val('c.predicted_states:y'), expected)
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_predict_comp_rk4_3seg(self):
        num_seg = 3
        state_options = {'y': {'shape': (1,), 'units': 'm'}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(num_seg, 4, 1), units='m')
        ivc.add_output('y0', shape=(num_seg, 1), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStatePredictComp(num_segments=num_seg, method='RK4',
                                                         state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states_per_seg:y')

        p.setup(check=True, force_alloc_complex=True)

        p['y0'] = np.array([[0.5, 1.425130208333333, 2.639602661132812]]).T
        p['k:y'] = np.array([[[0.75000000],
                              [0.90625000],
                              [0.94531250],
                              [1.09765625]],

                             [[1.087565104166667],
                              [1.203206380208333],
                              [1.232116699218750],
                              [1.328623453776042]],

                             [[1.319801330566406],
                              [1.368501663208008],
                              [1.380676746368408],
                              [1.385139703750610]]])

        p.run_model()

        p.model.list_outputs(print_arrays=True)

        expected = np.array([[0.50000000],
                             [0.87500000],
                             [0.953125],
                             [1.4453125],

                             [1.425130208333333],
                             [1.968912760416667],
                             [2.026733398437500],
                             [2.657246907552083],

                             [2.639602661132812],
                             [3.299503326416016],
                             [3.323853492736816],
                             [4.020279407501221]])

        assert_near_equal(p.get_val('c.predicted_states:y'), expected)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_predict_comp_rk4_vector(self):
        num_seg = 2
        state_options = {'y': {'shape': (2,), 'units': 'm'}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(num_seg, 4, 2), units='m')
        ivc.add_output('y0', shape=(num_seg, 2), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStatePredictComp(num_segments=num_seg, method='RK4',
                                                         state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states_per_seg:y')

        p.setup(check=True, force_alloc_complex=True)

        p['y0'] = [[0.500000000000000, 2.639602661132812],
                   [1.425130208333333, 4.006818970044454]]

        p['k:y'] = np.array([[[0.75000000, 1.319801330566406],
                              [0.90625000, 1.368501663208008],
                              [0.94531250, 1.380676746368408],
                              [1.09765625, 1.385139703750610]],

                             [[1.087565104166667, 1.378409485022227],
                              [1.203206380208333, 1.316761856277783],
                              [1.232116699218750, 1.301349949091673],
                              [1.328623453776042, 1.154084459568063]]])

        p.run_model()

        expected = np.array([[0.50000000, 2.639602661132812],
                             [0.87500000, 3.299503326416016],
                             [0.95312500, 3.323853492736816],
                             [1.44531250, 4.020279407501221],
                             [1.425130208333333, 4.006818970044454],
                             [1.968912760416667, 4.696023712555567],
                             [2.026733398437500, 4.665199898183346],
                             [2.657246907552083, 5.308168919136127]])

        assert_near_equal(p.get_val('c.predicted_states:y'), expected)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_advance_comp_rk4_matrix(self):
        state_options = {'y': {'shape': (2, 2), 'units': 'm'}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('k:y', shape=(1, 4, 2, 2), units='m')
        ivc.add_output('y0', shape=(1, 2, 2), units='m')

        p.model.add_subsystem('c',
                              RungeKuttaStatePredictComp(num_segments=1, method='RK4',
                                                         state_options=state_options))

        p.model.connect('k:y', 'c.k:y')
        p.model.connect('y0', 'c.initial_states_per_seg:y')

        p.setup(check=True, force_alloc_complex=True)

        p['y0'] = [[[0.5, 1.425130208333333],
                    [2.639602661132812, 4.006818970044454]]]

        p['k:y'] = np.array([[[[0.75, 1.087565104166667],
                              [1.319801330566406, 1.378409485022227]],

                             [[0.90625, 1.203206380208333],
                              [1.368501663208008,  1.316761856277783]],

                             [[0.9453125, 1.23211669921875],
                              [1.380676746368408,  1.301349949091673]],

                             [[1.09765625, 1.328623453776042],
                              [1.385139703750610,  1.154084459568063]]]])

        p.run_model()

        p.model.list_outputs(print_arrays=True)

        assert_near_equal(p.get_val('c.predicted_states:y'),
                          [[[0.5, 1.425130208333333],
                            [2.639602661132812, 4.006818970044454]],

                           [[0.875, 1.968912760416667],
                            [3.299503326416016, 4.696023712555567]],

                           [[0.953125, 2.0267333984375],
                            [3.323853492736816, 4.665199898183346]],

                           [[1.4453125, 2.657246907552083],
                            [4.020279407501221, 5.308168919136127]]])

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
