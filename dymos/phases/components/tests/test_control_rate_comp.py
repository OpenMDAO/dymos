from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.components import TimeComp
from dymos.phases.components import ControlRateComp
from dymos.phases.grid_data import GridData


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

    def test_control_rate_scalar_gl(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='gauss-lobatto')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (1,), 'dynamic': True},
                    'b': {'units': 'm', 'shape': (1,), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros(gd.subset_num_nodes['control_disc']), units='m')
        ivc.add_output('controls:b', val=np.zeros(gd.subset_num_nodes['control_disc']), units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp', subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('controls:b', 'control_rate_comp.controls:b')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'] = f_a(t)
        p['controls:b'] = f_b(t)

        p.run_model()

        a_rate_expected = f1_a(t)
        b_rate_expected = f1_b(t)

        a_rate2_expected = f2_a(t)
        b_rate2_expected = f2_b(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'],
                            np.atleast_2d(a_rate_expected).T)

        assert_almost_equal(p['control_rate_comp.control_rates:b_rate'],
                            np.atleast_2d(b_rate_expected).T)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'],
                            np.atleast_2d(a_rate2_expected).T)

        assert_almost_equal(p['control_rate_comp.control_rates:b_rate2'],
                            np.atleast_2d(b_rate2_expected).T)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, out_stream=None, method='cs')
        assert_check_partials(cpd)

    def test_control_rate_vector_gl(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='gauss-lobatto')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (3,), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_disc'], 3)),
                       units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'][:, 0] = f_a(t)
        p['controls:a'][:, 1] = f_b(t)
        p['controls:a'][:, 2] = f_c(t)

        p.run_model()

        a0_rate_expected = f1_a(t)
        a1_rate_expected = f1_b(t)
        a2_rate_expected = f1_c(t)

        a0_rate2_expected = f2_a(t)
        a1_rate2_expected = f2_b(t)
        a2_rate2_expected = f2_c(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0],
                            a0_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1],
                            a1_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 2],
                            a2_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1],
                            a1_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 2],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_control_rate_matrix_3x1_gl(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='gauss-lobatto')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (3, 1), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_disc'], 3, 1)),
                       units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'][:, 0, 0] = f_a(t)
        p['controls:a'][:, 1, 0] = f_b(t)
        p['controls:a'][:, 2, 0] = f_c(t)

        p.run_model()

        a0_rate_expected = f1_a(t)
        a1_rate_expected = f1_b(t)
        a2_rate_expected = f1_c(t)

        a0_rate2_expected = f2_a(t)
        a1_rate2_expected = f2_b(t)
        a2_rate2_expected = f2_c(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0, 0],
                            a0_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1, 0],
                            a1_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 2, 0],
                            a2_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1, 0],
                            a1_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 2, 0],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_control_rate_matrix_2x2_gl(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='gauss-lobatto')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (2, 2), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_disc'], 2, 2)),
                       units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp', subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'][:, 0, 0] = f_a(t)
        p['controls:a'][:, 0, 1] = f_b(t)
        p['controls:a'][:, 1, 0] = f_c(t)
        p['controls:a'][:, 1, 1] = f_d(t)

        p.run_model()

        a0_rate_expected = f1_a(t)
        a1_rate_expected = f1_b(t)
        a2_rate_expected = f1_c(t)
        a3_rate_expected = f1_d(t)

        a0_rate2_expected = f2_a(t)
        a1_rate2_expected = f2_b(t)
        a2_rate2_expected = f2_c(t)
        a3_rate2_expected = f2_d(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0, 0],
                            a0_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0, 1],
                            a1_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1, 0],
                            a2_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1, 1],
                            a3_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0, 1],
                            a1_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1, 0],
                            a2_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1, 1],
                            a3_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_control_rate_scalar_radau(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='radau-ps')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (1,), 'dynamic': True},
                    'b': {'units': 'm', 'shape': (1,), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros(gd.subset_num_nodes['control_disc']), units='m')
        ivc.add_output('controls:b', val=np.zeros(gd.subset_num_nodes['control_disc']), units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp', subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('controls:b', 'control_rate_comp.controls:b')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'] = f_a(t)
        p['controls:b'] = f_b(t)

        p.run_model()

        a_rate_expected = f1_a(t)
        b_rate_expected = f1_b(t)

        a_rate2_expected = f2_a(t)
        b_rate2_expected = f2_b(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'],
                            np.atleast_2d(a_rate_expected).T)

        assert_almost_equal(p['control_rate_comp.control_rates:b_rate'],
                            np.atleast_2d(b_rate_expected).T)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'],
                            np.atleast_2d(a_rate2_expected).T)

        assert_almost_equal(p['control_rate_comp.control_rates:b_rate2'],
                            np.atleast_2d(b_rate2_expected).T)

        np.set_printoptions(linewidth=1024)
        p.check_partials(compact_print=False, out_stream=None)

    def test_control_rate_vector_radau(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='radau-ps')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (3,), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_disc'], 3)),
                       units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'][:, 0] = f_a(t)
        p['controls:a'][:, 1] = f_b(t)
        p['controls:a'][:, 2] = f_c(t)

        p.run_model()

        a0_rate_expected = f1_a(t)
        a1_rate_expected = f1_b(t)
        a2_rate_expected = f1_c(t)

        a0_rate2_expected = f2_a(t)
        a1_rate2_expected = f2_b(t)
        a2_rate2_expected = f2_c(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0],
                            a0_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1],
                            a1_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 2],
                            a2_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1],
                            a1_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 2],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_control_rate_matrix_3x1_radau(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='radau-ps')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (3, 1), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_disc'], 3, 1)),
                       units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'][:, 0, 0] = f_a(t)
        p['controls:a'][:, 1, 0] = f_b(t)
        p['controls:a'][:, 2, 0] = f_c(t)

        p.run_model()

        a0_rate_expected = f1_a(t)
        a1_rate_expected = f1_b(t)
        a2_rate_expected = f1_c(t)

        a0_rate2_expected = f2_a(t)
        a1_rate2_expected = f2_b(t)
        a2_rate2_expected = f2_c(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0, 0],
                            a0_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1, 0],
                            a1_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 2, 0],
                            a2_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1, 0],
                            a1_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 2, 0],
                            a2_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, method='cs', out_stream=None)

        assert_check_partials(cpd)

    def test_control_rate_matrix_2x2_radau(self):

        segends = np.array([0.0, 3.0, 10.0])

        gd = GridData(num_segments=2,
                      transcription_order=5,
                      segment_ends=segends,
                      transcription='radau-ps')

        p = Problem(model=Group())

        controls = {'a': {'units': 'm', 'shape': (2, 2), 'dynamic': True}}

        ivc = IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('controls:a', val=np.zeros((gd.subset_num_nodes['control_disc'], 2, 2)),
                       units='m')
        ivc.add_output('t_initial', val=0.0, units='s')
        ivc.add_output('t_duration', val=10.0, units='s')

        p.model.add_subsystem('time_comp',
                              subsys=TimeComp(grid_data=gd, units='s'),
                              promotes_inputs=['t_initial', 't_duration'],
                              promotes_outputs=['time', 'dt_dstau'])

        p.model.add_subsystem('control_rate_comp',
                              subsys=ControlRateComp(grid_data=gd,
                                                     control_options=controls,
                                                     time_units='s'))

        p.model.connect('controls:a', 'control_rate_comp.controls:a')
        p.model.connect('dt_dstau', 'control_rate_comp.dt_dstau')

        p.setup(force_alloc_complex=True)

        p['t_initial'] = 0.0
        p['t_duration'] = 3.0

        p.run_model()

        t = p['time']
        p['controls:a'][:, 0, 0] = f_a(t)
        p['controls:a'][:, 0, 1] = f_b(t)
        p['controls:a'][:, 1, 0] = f_c(t)
        p['controls:a'][:, 1, 1] = f_d(t)

        p.run_model()

        a0_rate_expected = f1_a(t)
        a1_rate_expected = f1_b(t)
        a2_rate_expected = f1_c(t)
        a3_rate_expected = f1_d(t)

        a0_rate2_expected = f2_a(t)
        a1_rate2_expected = f2_b(t)
        a2_rate2_expected = f2_c(t)
        a3_rate2_expected = f2_d(t)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0, 0],
                            a0_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 0, 1],
                            a1_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1, 0],
                            a2_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate'][:, 1, 1],
                            a3_rate_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0, 0],
                            a0_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 0, 1],
                            a1_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1, 0],
                            a2_rate2_expected)

        assert_almost_equal(p['control_rate_comp.control_rates:a_rate2'][:, 1, 1],
                            a3_rate2_expected)

        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False, method='cs', out_stream=None)

        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
