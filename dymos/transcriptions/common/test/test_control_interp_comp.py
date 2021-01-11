import itertools
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.common import TimeComp
from dymos.transcriptions.common.control_group import ControlInterpComp
from dymos.transcriptions.grid_data import GridData

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig

TimeComp = CompWrapperConfig(TimeComp)
ControlInterpComp = CompWrapperConfig(ControlInterpComp)


# Test 1:  Let x = t**2, f = 2*t
def f_a(t):
    return t ** 2


def f1_a(t):
    return 2 * t


def f2_a(t):
    return 2.0 * np.ones_like(t)


# Test 1:  Let v = t**3-10*t**2, f = 3*t**2 - 20*t
def f_b(t):
    return t ** 3 - 10 * t ** 2


def f1_b(t):
    return 3 * t ** 2 - 20 * t


def f2_b(t):
    return 6 * t - 20


def f_c(t):
    return t ** 2


def f1_c(t):
    return 2 * t


def f2_c(t):
    return 2.0 * np.ones_like(t)


def f_d(t):
    return t ** 3


def f1_d(t):
    return 3 * t ** 2


def f2_d(t):
    return 6 * t


class TestControlRateComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_control_interp_scalar(self):
        param_list = itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                                       [True, False],  # compressed
                                       )
        for transcription, compressed in param_list:
            with self.subTest():
                segends = np.array([0.0, 3.0, 10.0])

                gd = GridData(num_segments=2,
                              transcription_order=5,
                              segment_ends=segends,
                              transcription=transcription,
                              compressed=compressed)

                p = om.Problem(model=om.Group())

                controls = {'a': {'units': 'm', 'shape': (1,), 'dynamic': True},
                            'b': {'units': 'm', 'shape': (1,), 'dynamic': True}}

                ivc = om.IndepVarComp()
                p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

                ivc.add_output('controls:a',
                               val=np.zeros((gd.subset_num_nodes['control_input'], 1)),
                               units='m')

                ivc.add_output('controls:b',
                               val=np.zeros((gd.subset_num_nodes['control_input'], 1)),
                               units='m')

                ivc.add_output('t_initial', val=0.0, units='s')
                ivc.add_output('t_duration', val=10.0, units='s')

                p.model.add_subsystem('time_comp',
                                      subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                                      node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                                      promotes_inputs=['t_initial', 't_duration'],
                                      promotes_outputs=['time', 'dt_dstau'])

                p.model.add_subsystem('control_interp_comp',
                                      subsys=ControlInterpComp(grid_data=gd,
                                                               control_options=controls,
                                                               time_units='s'),
                                      promotes_inputs=['controls:*'])

                p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

                p.setup(force_alloc_complex=True)

                p['t_initial'] = 0.0
                p['t_duration'] = 3.0

                p.run_model()

                t = p['time']
                p['controls:a'][:, 0] = f_a(t[gd.subset_node_indices['control_input']])
                p['controls:b'][:, 0] = f_b(t[gd.subset_node_indices['control_input']])

                p.run_model()

                a_value_expected = f_a(t)
                b_value_expected = f_b(t)

                a_rate_expected = f1_a(t)
                b_rate_expected = f1_b(t)

                a_rate2_expected = f2_a(t)
                b_rate2_expected = f2_b(t)

                assert_almost_equal(p['control_interp_comp.control_values:a'],
                                    np.atleast_2d(a_value_expected).T)

                assert_almost_equal(p['control_interp_comp.control_values:b'],
                                    np.atleast_2d(b_value_expected).T)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'],
                                    np.atleast_2d(a_rate_expected).T)

                assert_almost_equal(p['control_interp_comp.control_rates:b_rate'],
                                    np.atleast_2d(b_rate_expected).T)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'],
                                    np.atleast_2d(a_rate2_expected).T)

                assert_almost_equal(p['control_interp_comp.control_rates:b_rate2'],
                                    np.atleast_2d(b_rate2_expected).T)

                np.set_printoptions(linewidth=1024)
                cpd = p.check_partials(compact_print=False, method='cs')
                assert_check_partials(cpd)

    def test_control_interp_scalar_rk4(self, compressed=False):
        param_list = itertools.product([True, False],  # compressed
                                       )
        for compressed in param_list:
            with self.subTest():
                segends = np.array([0.0, 3.0, 9.0])

                gd = GridData(num_segments=2,
                              segment_ends=segends,
                              transcription='runge-kutta',
                              transcription_order='RK4',
                              compressed=compressed)

                p = om.Problem(model=om.Group())

                controls = {'a': {'units': 'm', 'shape': (1,), 'dynamic': True},
                            'b': {'units': 'm', 'shape': (1,), 'dynamic': True}}

                ivc = om.IndepVarComp()
                p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

                ivc.add_output('controls:a',
                               val=np.zeros((gd.subset_num_nodes['control_input'], 1)),
                               units='m')

                ivc.add_output('controls:b',
                               val=np.zeros((gd.subset_num_nodes['control_input'], 1)),
                               units='m')

                ivc.add_output('t_initial', val=0.0, units='s')
                ivc.add_output('t_duration', val=10.0, units='s')

                p.model.add_subsystem('time_comp',
                                      subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                                      node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                                      promotes_inputs=['t_initial', 't_duration'],
                                      promotes_outputs=['time', 'dt_dstau'])

                p.model.add_subsystem('control_interp_comp',
                                      subsys=ControlInterpComp(grid_data=gd,
                                                               control_options=controls,
                                                               time_units='s'),
                                      promotes_inputs=['controls:*'])

                p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

                p.setup(force_alloc_complex=True)

                p['t_initial'] = 0.0
                p['t_duration'] = 3.0

                p.run_model()

                t = p['time']
                p['controls:a'][:, 0] = f_a(t[gd.subset_node_indices['control_input']])
                p['controls:b'][:, 0] = f_b(t[gd.subset_node_indices['control_input']])

                p.run_model()

                a_value_expected = f_a(t)
                b_value_expected = f_b(t)

                a_rate_expected = f1_a(t)
                a_rate2_expected = f2_a(t)

                assert_almost_equal(p['control_interp_comp.control_values:a'],
                                    np.atleast_2d(a_value_expected).T)

                assert_almost_equal(p['control_interp_comp.control_values:b'],
                                    np.atleast_2d(b_value_expected).T)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'],
                                    np.atleast_2d(a_rate_expected).T)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'],
                                    np.atleast_2d(a_rate2_expected).T)

                np.set_printoptions(linewidth=1024)
                cpd = p.check_partials(compact_print=True, method='cs')
                assert_check_partials(cpd)

    def test_control_interp_vector(self, transcription='gauss-lobatto', compressed=True):
        param_list = itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                                       [True, False],  # compressed
                                       )
        for transcription, compressed in param_list:
            with self.subTest():
                segends = np.array([0.0, 3.0, 10.0])

                gd = GridData(num_segments=2,
                              transcription_order=5,
                              segment_ends=segends,
                              transcription=transcription,
                              compressed=compressed)

                p = om.Problem(model=om.Group())

                controls = {'a': {'units': 'm', 'shape': (3,), 'dynamic': True}}

                ivc = om.IndepVarComp()
                p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

                ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_input'], 3)),
                               units='m')
                ivc.add_output('t_initial', val=0.0, units='s')
                ivc.add_output('t_duration', val=10.0, units='s')

                p.model.add_subsystem('time_comp',
                                      subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                                      node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                                      promotes_inputs=['t_initial', 't_duration'],
                                      promotes_outputs=['time', 'dt_dstau'])

                p.model.add_subsystem('control_interp_comp',
                                      subsys=ControlInterpComp(grid_data=gd,
                                                               control_options=controls,
                                                               time_units='s'),
                                      promotes_inputs=['controls:*'])

                p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

                p.setup(force_alloc_complex=True)

                p['t_initial'] = 0.0
                p['t_duration'] = 3.0

                p.run_model()

                control_input_idxs = gd.subset_node_indices['control_input']

                t = p['time']
                p['controls:a'][:, 0] = f_a(t[control_input_idxs])
                p['controls:a'][:, 1] = f_b(t[control_input_idxs])
                p['controls:a'][:, 2] = f_c(t[control_input_idxs])

                p.run_model()

                a0_value_expected = f_a(t)
                a1_value_expected = f_b(t)
                a2_value_expected = f_c(t)

                a0_rate_expected = f1_a(t)
                a1_rate_expected = f1_b(t)
                a2_rate_expected = f1_c(t)

                a0_rate2_expected = f2_a(t)
                a1_rate2_expected = f2_b(t)
                a2_rate2_expected = f2_c(t)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 0],
                                    a0_value_expected)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 1],
                                    a1_value_expected)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 2],
                                    a2_value_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 0],
                                    a0_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 1],
                                    a1_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 2],
                                    a2_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 0],
                                    a0_rate2_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 1],
                                    a1_rate2_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 2],
                                    a2_rate2_expected)

                np.set_printoptions(linewidth=1024)
                cpd = p.check_partials(compact_print=True, method='cs')

                assert_check_partials(cpd)

    def test_control_interp_matrix_3x1(self, transcription='gauss-lobatto', compressed=True):
        param_list = itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                                       [True, False],  # compressed
                                       )
        for transcription, compressed in param_list:
            with self.subTest():
                segends = np.array([0.0, 3.0, 10.0])

                gd = GridData(num_segments=2,
                              transcription_order=5,
                              segment_ends=segends,
                              transcription=transcription,
                              compressed=compressed)

                p = om.Problem(model=om.Group())

                controls = {'a': {'units': 'm', 'shape': (3, 1), 'dynamic': True}}

                ivc = om.IndepVarComp()
                p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

                ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_input'], 3, 1)),
                               units='m')
                ivc.add_output('t_initial', val=0.0, units='s')
                ivc.add_output('t_duration', val=10.0, units='s')

                p.model.add_subsystem('time_comp',
                                      subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                                      node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                                      promotes_inputs=['t_initial', 't_duration'],
                                      promotes_outputs=['time', 'dt_dstau'])

                p.model.add_subsystem('control_interp_comp',
                                      subsys=ControlInterpComp(grid_data=gd,
                                                               control_options=controls,
                                                               time_units='s'),
                                      promotes_inputs=['controls:*'])

                p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

                p.setup(force_alloc_complex=True)

                p['t_initial'] = 0.0
                p['t_duration'] = 3.0

                p.run_model()

                t = p['time']
                control_input_idxs = gd.subset_node_indices['control_input']
                p['controls:a'][:, 0, 0] = f_a(t[control_input_idxs])
                p['controls:a'][:, 1, 0] = f_b(t[control_input_idxs])
                p['controls:a'][:, 2, 0] = f_c(t[control_input_idxs])

                p.run_model()

                a0_value_expected = f_a(t)
                a1_value_expected = f_b(t)
                a2_value_expected = f_c(t)

                a0_rate_expected = f1_a(t)
                a1_rate_expected = f1_b(t)
                a2_rate_expected = f1_c(t)

                a0_rate2_expected = f2_a(t)
                a1_rate2_expected = f2_b(t)
                a2_rate2_expected = f2_c(t)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 0, 0],
                                    a0_value_expected)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 1, 0],
                                    a1_value_expected)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 2, 0],
                                    a2_value_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 0, 0],
                                    a0_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 1, 0],
                                    a1_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 2, 0],
                                    a2_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 0, 0],
                                    a0_rate2_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 1, 0],
                                    a1_rate2_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 2, 0],
                                    a2_rate2_expected)

                np.set_printoptions(linewidth=1024)
                cpd = p.check_partials(compact_print=True, method='cs')

                assert_check_partials(cpd)

    def test_control_interp_matrix_2x2(self, transcription='gauss-lobatto', compressed=True):
        param_list = itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                                       [True, False],  # compressed
                                       )
        for transcription, compressed in param_list:
            with self.subTest():
                segends = np.array([0.0, 3.0, 10.0])

                gd = GridData(num_segments=2,
                              transcription_order=5,
                              segment_ends=segends,
                              transcription=transcription,
                              compressed=compressed)

                p = om.Problem(model=om.Group())

                controls = {'a': {'units': 'm', 'shape': (2, 2), 'dynamic': True}}

                ivc = om.IndepVarComp()
                p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

                ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_input'], 2, 2)),
                               units='m')
                ivc.add_output('t_initial', val=0.0, units='s')
                ivc.add_output('t_duration', val=10.0, units='s')

                p.model.add_subsystem('time_comp',
                                      subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                                      node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                                      promotes_inputs=['t_initial', 't_duration'],
                                      promotes_outputs=['time', 'dt_dstau'])

                p.model.add_subsystem('control_interp_comp',
                                      subsys=ControlInterpComp(grid_data=gd,
                                                               control_options=controls,
                                                               time_units='s'),
                                      promotes_inputs=['controls:*'])

                p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

                p.setup(force_alloc_complex=True)

                p['t_initial'] = 0.0
                p['t_duration'] = 3.0

                p.run_model()

                t = p['time']
                control_input_idxs = gd.subset_node_indices['control_input']
                p['controls:a'][:, 0, 0] = f_a(t[control_input_idxs])
                p['controls:a'][:, 0, 1] = f_b(t[control_input_idxs])
                p['controls:a'][:, 1, 0] = f_c(t[control_input_idxs])
                p['controls:a'][:, 1, 1] = f_d(t[control_input_idxs])

                p.run_model()

                a0_value_expected = f_a(t)
                a1_value_expected = f_b(t)
                a2_value_expected = f_c(t)
                a3_value_expected = f_d(t)

                a0_rate_expected = f1_a(t)
                a1_rate_expected = f1_b(t)
                a2_rate_expected = f1_c(t)
                a3_rate_expected = f1_d(t)

                a0_rate2_expected = f2_a(t)
                a1_rate2_expected = f2_b(t)
                a2_rate2_expected = f2_c(t)
                a3_rate2_expected = f2_d(t)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 0, 0],
                                    a0_value_expected)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 0, 1],
                                    a1_value_expected)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 1, 0],
                                    a2_value_expected)

                assert_almost_equal(p['control_interp_comp.control_values:a'][:, 1, 1],
                                    a3_value_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 0, 0],
                                    a0_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 0, 1],
                                    a1_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 1, 0],
                                    a2_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate'][:, 1, 1],
                                    a3_rate_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 0, 0],
                                    a0_rate2_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 0, 1],
                                    a1_rate2_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 1, 0],
                                    a2_rate2_expected)

                assert_almost_equal(p['control_interp_comp.control_rates:a_rate2'][:, 1, 1],
                                    a3_rate2_expected)

                with np.printoptions(linewidth=100000, edgeitems=100000):
                    cpd = p.check_partials(compact_print=True, method='cs')

                assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
