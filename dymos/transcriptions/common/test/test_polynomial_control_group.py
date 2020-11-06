import unittest

import numpy as np
from numpy.testing import assert_almost_equal
import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

from dymos.transcriptions.common import TimeComp, PolynomialControlGroup
from dymos.transcriptions.grid_data import GridData
from dymos.phase.options import PolynomialControlOptionsDictionary
from dymos.utils.lgl import lgl

from dymos.utils.misc import CompWrapperConfig, GroupWrapperConfig
TimeComp = CompWrapperConfig(TimeComp)
PolynomialControlGroup = GroupWrapperConfig(PolynomialControlGroup)


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


class TestInterpolatedControLGroup(unittest.TestCase):

    def test_polynomial_control_group_scalar_gl(self):
        transcription = 'gauss-lobatto'
        compressed = True

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary(),
                    'b': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['shape'] = (1, )
        controls['a']['opt'] = True

        controls['b']['units'] = 'm'
        controls['b']['order'] = 3
        controls['b']['shape'] = (1, )
        controls['b']['opt'] = True

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0] = f_a(t_control_input)
        p['polynomial_controls:b'][:, 0] = f_b(t_control_input)

        p.run_model()

        a_value_expected = f_a(t_all)
        b_value_expected = f_b(t_all)

        a_rate_expected = f1_a(t_all)
        b_rate_expected = f1_b(t_all)

        a_rate2_expected = f2_a(t_all)
        b_rate2_expected = f2_b(t_all)

        assert_almost_equal(p['polynomial_control_values:a'],
                            np.atleast_2d(a_value_expected).T)

        assert_almost_equal(p['polynomial_control_values:b'],
                            np.atleast_2d(b_value_expected).T)

        assert_almost_equal(p['polynomial_control_rates:a_rate'],
                            np.atleast_2d(a_rate_expected).T)

        assert_almost_equal(p['polynomial_control_rates:b_rate'],
                            np.atleast_2d(b_rate_expected).T)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'],
                            np.atleast_2d(a_rate2_expected).T)

        assert_almost_equal(p['polynomial_control_rates:b_rate2'],
                            np.atleast_2d(b_rate2_expected).T)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, out_stream=None, method='cs')
        assert_check_partials(cpd)

    def test_polynomial_control_group_scalar_radau(self):
        transcription = 'radau-ps'
        compressed = False

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary(),
                    'b': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['shape'] = (1, )
        controls['a']['opt'] = True

        controls['b']['units'] = 'm'
        controls['b']['order'] = 3
        controls['b']['shape'] = (1, )
        controls['b']['opt'] = True

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        # p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0] = f_a(t_control_input)
        p['polynomial_controls:b'][:, 0] = f_b(t_control_input)

        p.run_model()

        a_value_expected = f_a(t_all)
        b_value_expected = f_b(t_all)

        a_rate_expected = f1_a(t_all)
        b_rate_expected = f1_b(t_all)

        a_rate2_expected = f2_a(t_all)
        b_rate2_expected = f2_b(t_all)

        assert_almost_equal(p['polynomial_control_values:a'],
                            np.atleast_2d(a_value_expected).T)

        assert_almost_equal(p['polynomial_control_values:b'],
                            np.atleast_2d(b_value_expected).T)

        assert_almost_equal(p['polynomial_control_rates:a_rate'],
                            np.atleast_2d(a_rate_expected).T)

        assert_almost_equal(p['polynomial_control_rates:b_rate'],
                            np.atleast_2d(b_rate_expected).T)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'],
                            np.atleast_2d(a_rate2_expected).T)

        assert_almost_equal(p['polynomial_control_rates:b_rate2'],
                            np.atleast_2d(b_rate2_expected).T)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, out_stream=None, method='cs')
        assert_check_partials(cpd)

    def test_polynomial_control_group_scalar_rungekutta(self):
        transcription = 'runge-kutta'
        compressed = False

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order='RK4',
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary(),
                    'b': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['shape'] = (1, )
        controls['a']['opt'] = True

        controls['b']['units'] = 'm'
        controls['b']['order'] = 3
        controls['b']['shape'] = (1, )
        controls['b']['opt'] = True

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0] = f_a(t_control_input)
        p['polynomial_controls:b'][:, 0] = f_b(t_control_input)

        p.run_model()

        a_value_expected = f_a(t_all)
        b_value_expected = f_b(t_all)

        a_rate_expected = f1_a(t_all)
        b_rate_expected = f1_b(t_all)

        a_rate2_expected = f2_a(t_all)
        b_rate2_expected = f2_b(t_all)

        assert_almost_equal(p['polynomial_control_values:a'],
                            np.atleast_2d(a_value_expected).T)

        assert_almost_equal(p['polynomial_control_values:b'],
                            np.atleast_2d(b_value_expected).T)

        assert_almost_equal(p['polynomial_control_rates:a_rate'],
                            np.atleast_2d(a_rate_expected).T)

        assert_almost_equal(p['polynomial_control_rates:b_rate'],
                            np.atleast_2d(b_rate_expected).T)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'],
                            np.atleast_2d(a_rate2_expected).T)

        assert_almost_equal(p['polynomial_control_rates:b_rate2'],
                            np.atleast_2d(b_rate2_expected).T)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, out_stream=None, method='cs')
        assert_check_partials(cpd)

    def test_polynomial_control_group_vector_gl(self):
        transcription = 'gauss-lobatto'
        compressed = True

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['opt'] = True
        controls['a']['shape'] = (3,)

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        # p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0] = f_a(t_control_input)
        p['polynomial_controls:a'][:, 1] = f_b(t_control_input)
        p['polynomial_controls:a'][:, 2] = f_c(t_control_input)

        p.run_model()

        a0_value_expected = f_a(t_all)
        a1_value_expected = f_b(t_all)
        a2_value_expected = f_c(t_all)

        a0_rate_expected = f1_a(t_all)
        a1_rate_expected = f1_b(t_all)
        a2_rate_expected = f1_c(t_all)

        a0_rate2_expected = f2_a(t_all)
        a1_rate2_expected = f2_b(t_all)
        a2_rate2_expected = f2_c(t_all)

        assert_almost_equal(p['polynomial_control_values:a'][:, 0],
                            a0_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 1],
                            a1_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 2],
                            a2_value_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 0],
                            a0_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 1],
                            a1_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 2],
                            a2_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 1],
                            a1_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 2],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_polynomial_control_group_vector_radau(self):
        transcription = 'radau-ps'
        compressed = True

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['opt'] = True
        controls['a']['shape'] = (3,)

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        # p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0] = f_a(t_control_input)
        p['polynomial_controls:a'][:, 1] = f_b(t_control_input)
        p['polynomial_controls:a'][:, 2] = f_c(t_control_input)

        p.run_model()

        a0_value_expected = f_a(t_all)
        a1_value_expected = f_b(t_all)
        a2_value_expected = f_c(t_all)

        a0_rate_expected = f1_a(t_all)
        a1_rate_expected = f1_b(t_all)
        a2_rate_expected = f1_c(t_all)

        a0_rate2_expected = f2_a(t_all)
        a1_rate2_expected = f2_b(t_all)
        a2_rate2_expected = f2_c(t_all)

        assert_almost_equal(p['polynomial_control_values:a'][:, 0],
                            a0_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 1],
                            a1_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 2],
                            a2_value_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 0],
                            a0_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 1],
                            a1_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 2],
                            a2_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 1],
                            a1_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 2],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_polynomial_control_group_vector_rungekutta(self):
        transcription = 'runge-kutta'
        compressed = True

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order='RK4',
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['opt'] = True
        controls['a']['shape'] = (3,)

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        # p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0] = f_a(t_control_input)
        p['polynomial_controls:a'][:, 1] = f_b(t_control_input)
        p['polynomial_controls:a'][:, 2] = f_c(t_control_input)

        p.run_model()

        a0_value_expected = f_a(t_all)
        a1_value_expected = f_b(t_all)
        a2_value_expected = f_c(t_all)

        a0_rate_expected = f1_a(t_all)
        a1_rate_expected = f1_b(t_all)
        a2_rate_expected = f1_c(t_all)

        a0_rate2_expected = f2_a(t_all)
        a1_rate2_expected = f2_b(t_all)
        a2_rate2_expected = f2_c(t_all)

        assert_almost_equal(p['polynomial_control_values:a'][:, 0],
                            a0_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 1],
                            a1_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 2],
                            a2_value_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 0],
                            a0_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 1],
                            a1_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 2],
                            a2_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 1],
                            a1_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 2],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_polynomial_control_group_matrix_gl(self):
        transcription = 'gauss-lobatto'
        compressed = True

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['opt'] = True
        controls['a']['shape'] = (3, 1)

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        # p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0, 0] = f_a(t_control_input)
        p['polynomial_controls:a'][:, 1, 0] = f_b(t_control_input)
        p['polynomial_controls:a'][:, 2, 0] = f_c(t_control_input)

        p.run_model()

        a0_value_expected = f_a(t_all)
        a1_value_expected = f_b(t_all)
        a2_value_expected = f_c(t_all)

        a0_rate_expected = f1_a(t_all)
        a1_rate_expected = f1_b(t_all)
        a2_rate_expected = f1_c(t_all)

        a0_rate2_expected = f2_a(t_all)
        a1_rate2_expected = f2_b(t_all)
        a2_rate2_expected = f2_c(t_all)

        assert_almost_equal(p['polynomial_control_values:a'][:, 0, 0],
                            a0_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 1, 0],
                            a1_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 2, 0],
                            a2_value_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 0, 0],
                            a0_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 1, 0],
                            a1_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 2, 0],
                            a2_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 0, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 1, 0],
                            a1_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 2, 0],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_polynomial_control_group_matrix_radau(self):
        transcription = 'radau-ps'
        compressed = True

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['opt'] = True
        controls['a']['shape'] = (3, 1)

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        # p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0, 0] = f_a(t_control_input)
        p['polynomial_controls:a'][:, 1, 0] = f_b(t_control_input)
        p['polynomial_controls:a'][:, 2, 0] = f_c(t_control_input)

        p.run_model()

        a0_value_expected = f_a(t_all)
        a1_value_expected = f_b(t_all)
        a2_value_expected = f_c(t_all)

        a0_rate_expected = f1_a(t_all)
        a1_rate_expected = f1_b(t_all)
        a2_rate_expected = f1_c(t_all)

        a0_rate2_expected = f2_a(t_all)
        a1_rate2_expected = f2_b(t_all)
        a2_rate2_expected = f2_c(t_all)

        assert_almost_equal(p['polynomial_control_values:a'][:, 0, 0],
                            a0_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 1, 0],
                            a1_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 2, 0],
                            a2_value_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 0, 0],
                            a0_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 1, 0],
                            a1_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 2, 0],
                            a2_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 0, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 1, 0],
                            a1_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 2, 0],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_polynomial_control_group_matrix_rungekutta(self):
        transcription = 'runge-kutta'
        compressed = True

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order='RK4',
                      segment_ends=segends,
                      transcription=transcription,
                      compressed=compressed)

        p = om.Problem(model=om.Group())

        controls = {'a': PolynomialControlOptionsDictionary()}

        controls['a']['units'] = 'm'
        controls['a']['order'] = 3
        controls['a']['opt'] = True
        controls['a']['shape'] = (3, 1)

        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(num_nodes=gd.num_nodes, node_ptau=gd.node_ptau,
                                              node_dptau_dstau=gd.node_dptau_dstau, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        polynomial_control_group = PolynomialControlGroup(grid_data=gd,
                                                          polynomial_control_options=controls,
                                                          time_units='s')

        p.model.add_subsystem('polynomial_control_group',
                              subsys=polynomial_control_group,
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        # p.model.connect('dt_dstau', 'control_interp_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        control_nodes_ptau, _ = lgl(controls['a']['order'] + 1)

        t_control_input = p['t_initial'] + 0.5 * (control_nodes_ptau + 1) * p['t_duration']
        t_all = p['time']

        p['polynomial_controls:a'][:, 0, 0] = f_a(t_control_input)
        p['polynomial_controls:a'][:, 1, 0] = f_b(t_control_input)
        p['polynomial_controls:a'][:, 2, 0] = f_c(t_control_input)

        p.run_model()

        a0_value_expected = f_a(t_all)
        a1_value_expected = f_b(t_all)
        a2_value_expected = f_c(t_all)

        a0_rate_expected = f1_a(t_all)
        a1_rate_expected = f1_b(t_all)
        a2_rate_expected = f1_c(t_all)

        a0_rate2_expected = f2_a(t_all)
        a1_rate2_expected = f2_b(t_all)
        a2_rate2_expected = f2_c(t_all)

        assert_almost_equal(p['polynomial_control_values:a'][:, 0, 0],
                            a0_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 1, 0],
                            a1_value_expected)

        assert_almost_equal(p['polynomial_control_values:a'][:, 2, 0],
                            a2_value_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 0, 0],
                            a0_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 1, 0],
                            a1_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate'][:, 2, 0],
                            a2_rate_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 0, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 1, 0],
                            a1_rate2_expected)

        assert_almost_equal(p['polynomial_control_rates:a_rate2'][:, 2, 0],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
