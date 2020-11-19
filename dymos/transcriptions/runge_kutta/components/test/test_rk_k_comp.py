import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.runge_kutta.components.runge_kutta_k_comp import RungeKuttaKComp

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
RungeKuttaKComp = CompWrapperConfig(RungeKuttaKComp)


class TestRKKComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_rk_k_comp_rk4_scalar(self):
        state_options = {'y': {'shape': (1,), 'units': 'm'}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('h', shape=(4,), units='s')
        ivc.add_output('f:y', shape=(4, 4, 1), units='m/s')

        p.model.add_subsystem('c', RungeKuttaKComp(num_segments=4, method='RK4',
                                                   state_options=state_options, time_units='s'))

        p.model.connect('f:y', 'c.f:y')
        p.model.connect('h', 'c.h')

        p.setup(check=True, force_alloc_complex=True)

        p['h'] = [0.5, 0.5, 0.5, 0.5]
        p['f:y'] = np.array([[[1.50000000],
                              [1.81250000],
                              [1.89062500],
                              [2.19531250]],

                             [[2.17513021],
                              [2.40641276],
                              [2.46423340],
                              [2.65724691]],

                             [[2.63960266],
                              [2.73700333],
                              [2.76135349],
                              [2.77027941]],

                             [[2.75681897],
                              [2.63352371],
                              [2.60269990],
                              [2.30816892]]])

        np.set_printoptions(linewidth=1024)

        p.run_model()

        expected = np.array([[[0.75000000],
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

        assert_near_equal(p.get_val('c.k:y'), expected, tolerance=1.0E-9)

        p.model.list_outputs(print_arrays=True)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_advance_comp_rk4_vector(self):
        num_seg = 2
        state_options = {'y': {'shape': (num_seg,), 'units': 'm'}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('h', shape=(num_seg,), units='s')
        ivc.add_output('f:y', shape=(num_seg, 4, 2), units='m/s')

        p.model.add_subsystem('c', RungeKuttaKComp(num_segments=num_seg, method='RK4',
                                                   state_options=state_options, time_units='s'))

        p.model.connect('f:y', 'c.f:y')
        p.model.connect('h', 'c.h')

        p.setup(check=True, force_alloc_complex=True)

        p['h'] = [0.5, 0.5]
        p['f:y'] = np.array([[[1.50000000, 2.63960266],
                              [1.81250000, 2.73700333],
                              [1.89062500, 2.76135349],
                              [2.19531250, 2.77027941]],

                             [[2.17513021, 2.75681897],
                              [2.40641276, 2.63352371],
                              [2.46423340, 2.60269990],
                              [2.65724691, 2.30816892]]])

        p.run_model()

        expected = np.array([[[0.75000000, 1.319801330566406],
                              [0.90625000, 1.368501663208008],
                              [0.94531250, 1.380676746368408],
                              [1.09765625, 1.385139703750610]],

                             [[1.087565104166667, 1.378409485022227],
                              [1.203206380208333, 1.316761856277783],
                              [1.232116699218750, 1.301349949091673],
                              [1.328623453776042, 1.154084459568063]]])

        assert_near_equal(p.get_val('c.k:y'), expected, tolerance=1.0E-9)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_state_advance_comp_rk4_matrix(self):
        num_seg = 1
        num_stages = 4
        state_options = {'y': {'shape': (2, 2), 'units': 'm'}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('h', shape=(num_seg,), units='s')
        ivc.add_output('f:y', shape=(num_seg, num_stages, 2, 2), units='m/s')

        p.model.add_subsystem('c', RungeKuttaKComp(num_segments=num_seg, method='RK4',
                                                   state_options=state_options, time_units='s'))

        p.model.connect('f:y', 'c.f:y')
        p.model.connect('h', 'c.h')

        p.setup(check=True, force_alloc_complex=True)

        p['h'] = [0.5]
        p['f:y'] = np.array([[[[1.50000000, 2.63960266],
                               [2.17513021, 2.75681897]],

                              [[1.81250000, 2.73700333],
                               [2.40641276, 2.63352371]],

                              [[1.89062500, 2.76135349],
                               [2.46423340, 2.60269990]],

                              [[2.19531250, 2.77027941],
                               [2.65724691, 2.30816892]]]])

        p.run_model()

        expected = np.array([[[[0.75000000, 1.319801330566406],
                               [1.087565104166667, 1.378409485022227]],

                              [[0.90625000, 1.368501663208008],
                               [1.203206380208333, 1.316761856277783]],

                              [[0.94531250, 1.380676746368408],
                               [1.232116699218750, 1.301349949091673]],

                              [[1.09765625, 1.385139703750610],
                               [1.328623453776042, 1.154084459568063]]]])

        assert_near_equal(p.get_val('c.k:y'), expected, tolerance=1.0E-9)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
