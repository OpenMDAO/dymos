from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.phases.components import GaussLobattoPathConstraintComp, RadauPathConstraintComp, \
    ExplicitPathConstraintComp
from dymos.phases.grid_data import GridData
from dymos.phases.options import ControlOptionsDictionary


class TestPathConstraintCompGL(unittest.TestCase):

    def setUp(self):

        transcription = 'gauss-lobatto'

        self.gd = gd = GridData(num_segments=2,
                                transcription_order=3,
                                segment_ends=[0.0, 3.0, 10.0],
                                transcription=transcription)

        ndn = gd.subset_num_nodes['disc']
        ncn = gd.subset_num_nodes['col']
        nn = ndn + ncn

        self.p = Problem(model=Group())

        controls = {'a': ControlOptionsDictionary(),
                    'b': ControlOptionsDictionary(),
                    'c': ControlOptionsDictionary(),
                    'd': ControlOptionsDictionary()}

        controls['a'].update({'units': 'm', 'shape': (1,), 'opt': False})
        controls['b'].update({'units': 's', 'shape': (3,), 'opt': False})
        controls['c'].update({'units': 'kg', 'shape': (3, 3), 'opt': False})

        ivc = IndepVarComp()
        self.p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('a_disc', val=np.zeros((ndn, 1)), units='m')
        ivc.add_output('a_col', val=np.zeros((ncn, 1)), units='m')
        ivc.add_output('b_disc', val=np.zeros((ndn, 3)), units='s')
        ivc.add_output('b_col', val=np.zeros((ncn, 3)), units='s')
        ivc.add_output('c_disc', val=np.zeros((ndn, 3, 3)), units='kg')
        ivc.add_output('c_col', val=np.zeros((ncn, 3, 3)), units='kg')
        ivc.add_output('a_ctrl', val=np.zeros((nn, 1)), units='m')
        ivc.add_output('b_ctrl', val=np.zeros((nn, 3)), units='s')
        ivc.add_output('c_ctrl', val=np.zeros((nn, 3, 3)), units='kg')

        path_comp = GaussLobattoPathConstraintComp(grid_data=gd)

        self.p.model.add_subsystem('path_constraints', subsys=path_comp)

        path_comp._add_path_constraint('a', var_class='ode',
                                       shape=(1,), lower=0, upper=10, units='m')
        path_comp._add_path_constraint('b', var_class='ode',
                                       shape=(3,), lower=0, upper=10, units='s')
        path_comp._add_path_constraint('c', var_class='ode',
                                       shape=(3, 3), lower=0, upper=10, units='kg')

        path_comp._add_path_constraint('a_ctrl', var_class='time',
                                       shape=(1,), lower=0, upper=10, units='m')
        path_comp._add_path_constraint('b_ctrl', var_class='indep_control',
                                       shape=(3,), lower=0, upper=10, units='s')
        path_comp._add_path_constraint('c_ctrl', var_class='control_rate',
                                       shape=(3, 3), lower=0, upper=10, units='kg')

        self.p.model.connect('a_disc', 'path_constraints.disc_values:a')
        self.p.model.connect('a_col', 'path_constraints.col_values:a')

        self.p.model.connect('b_disc', 'path_constraints.disc_values:b')
        self.p.model.connect('b_col', 'path_constraints.col_values:b')

        self.p.model.connect('c_disc', 'path_constraints.disc_values:c')
        self.p.model.connect('c_col', 'path_constraints.col_values:c')

        self.p.model.connect('a_ctrl', 'path_constraints.all_values:a_ctrl')
        self.p.model.connect('b_ctrl', 'path_constraints.all_values:b_ctrl')
        self.p.model.connect('c_ctrl', 'path_constraints.all_values:c_ctrl')

        self.p.setup()

        self.p['a_disc'] = np.random.rand(*self.p['a_disc'].shape)
        self.p['a_col'] = np.random.rand(*self.p['a_col'].shape)

        self.p['b_disc'] = np.random.rand(*self.p['b_disc'].shape)
        self.p['b_col'] = np.random.rand(*self.p['b_col'].shape)

        self.p['c_disc'] = np.random.rand(*self.p['c_disc'].shape)
        self.p['c_col'] = np.random.rand(*self.p['c_col'].shape)

        self.p['a_ctrl'] = np.random.rand(*self.p['a_ctrl'].shape)
        self.p['b_ctrl'] = np.random.rand(*self.p['b_ctrl'].shape)
        self.p['c_ctrl'] = np.random.rand(*self.p['c_ctrl'].shape)

        self.p.run_model()

    def test_results(self):
        p = self.p
        gd = self.gd
        assert_almost_equal(p['a_disc'],
                            p['path_constraints.path:a'][gd.subset_node_indices['disc'], ...])

        assert_almost_equal(p['a_col'],
                            p['path_constraints.path:a'][gd.subset_node_indices['col'], ...])

        assert_almost_equal(p['b_disc'],
                            p['path_constraints.path:b'][gd.subset_node_indices['disc'], ...])

        assert_almost_equal(p['b_col'],
                            p['path_constraints.path:b'][gd.subset_node_indices['col'], ...])

        assert_almost_equal(p['c_disc'],
                            p['path_constraints.path:c'][gd.subset_node_indices['disc'], ...])

        assert_almost_equal(p['c_col'],
                            p['path_constraints.path:c'][gd.subset_node_indices['col'], ...])

        assert_almost_equal(p['a_ctrl'],
                            p['path_constraints.path:a_ctrl'])

        assert_almost_equal(p['b_ctrl'],
                            p['path_constraints.path:b_ctrl'])

        assert_almost_equal(p['c_ctrl'],
                            p['path_constraints.path:c_ctrl'])

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)


class TestPathConstraintCompRadau(unittest.TestCase):

    def setUp(self):

        transcription = 'radau-ps'

        self.gd = gd = GridData(num_segments=2,
                                transcription_order=3,
                                segment_ends=[0.0, 3.0, 10.0],
                                transcription=transcription)

        ndn = gd.subset_num_nodes['disc']

        self.p = Problem(model=Group())

        controls = {'a': ControlOptionsDictionary(),
                    'b': ControlOptionsDictionary(),
                    'c': ControlOptionsDictionary(),
                    'd': ControlOptionsDictionary()}

        controls['a'].update({'units': 'm', 'shape': (1,), 'opt': False})
        controls['b'].update({'units': 's', 'shape': (3,), 'opt': False})
        controls['c'].update({'units': 'kg', 'shape': (3, 3), 'opt': False})

        ivc = IndepVarComp()
        self.p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('a_disc', val=np.zeros((ndn, 1)), units='m')
        ivc.add_output('b_disc', val=np.zeros((ndn, 3)), units='s')
        ivc.add_output('c_disc', val=np.zeros((ndn, 3, 3)), units='kg')

        path_comp = RadauPathConstraintComp(grid_data=gd)

        self.p.model.add_subsystem('path_constraints', subsys=path_comp)

        path_comp._add_path_constraint('a', var_class='ode', shape=(1,),
                                       lower=0, upper=10, units='m')
        path_comp._add_path_constraint('b', var_class='input_control', shape=(3,),
                                       lower=0, upper=10, units='s')
        path_comp._add_path_constraint('c', var_class='control_rate2', shape=(3, 3),
                                       lower=0, upper=10, units='kg')

        self.p.model.connect('a_disc', 'path_constraints.all_values:a')
        self.p.model.connect('b_disc', 'path_constraints.all_values:b')
        self.p.model.connect('c_disc', 'path_constraints.all_values:c')

        self.p.setup()

        self.p.run_model()

    def test_results(self):
        p = self.p
        gd = self.gd
        assert_almost_equal(p['a_disc'],
                            p['path_constraints.path:a'][gd.subset_node_indices['state_disc'], ...])

        assert_almost_equal(p['b_disc'],
                            p['path_constraints.path:b'][gd.subset_node_indices['state_disc'], ...])

        assert_almost_equal(p['c_disc'],
                            p['path_constraints.path:c'][gd.subset_node_indices['state_disc'], ...])

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)


class TestPathConstraintCompExplicit(unittest.TestCase):

    def setUp(self):

        transcription = 'explicit'

        self.gd = gd = GridData(num_segments=2,
                                transcription_order=3,
                                num_steps_per_segment=[4, 4],
                                segment_ends=[0.0, 3.0, 10.0],
                                transcription=transcription)

        self.p = Problem(model=Group())

        ivc = IndepVarComp()
        self.p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        ivc.add_output('seg_0:a', val=np.zeros((gd.num_steps_per_segment[0]+1, 1)), units='m')
        ivc.add_output('seg_1:a', val=np.zeros((gd.num_steps_per_segment[1]+1, 1)), units='m')

        ivc.add_output('seg_0:b', val=np.zeros((gd.num_steps_per_segment[0]+1, 3)), units='s')
        ivc.add_output('seg_1:b', val=np.zeros((gd.num_steps_per_segment[1]+1, 3)), units='s')

        ivc.add_output('seg_0:c', val=np.zeros((gd.num_steps_per_segment[0]+1, 3, 3)), units='N*m')
        ivc.add_output('seg_1:c', val=np.zeros((gd.num_steps_per_segment[1]+1, 3, 3)), units='N*m')

        path_comp = ExplicitPathConstraintComp(grid_data=gd)

        self.p.model.add_subsystem('path_constraints', subsys=path_comp)

        path_comp._add_path_constraint('a', var_class='ode', shape=(1,),
                                       lower=0, upper=10, units='m')
        path_comp._add_path_constraint('b', var_class='input_control', shape=(3,),
                                       lower=0, upper=10, units='s')
        path_comp._add_path_constraint('c', var_class='control_rate2', shape=(3, 3),
                                       lower=0, upper=10, units='N*m')

        self.p.model.connect('seg_0:a', 'path_constraints.seg_0_values:a')
        self.p.model.connect('seg_1:a', 'path_constraints.seg_1_values:a')

        self.p.model.connect('seg_0:b', 'path_constraints.seg_0_values:b')
        self.p.model.connect('seg_1:b', 'path_constraints.seg_1_values:b')

        self.p.model.connect('seg_0:c', 'path_constraints.seg_0_values:c')
        self.p.model.connect('seg_1:c', 'path_constraints.seg_1_values:c')

        self.p.setup()

        self.p.set_val('seg_0:a', np.reshape(100 + np.arange(5), (5, 1)))
        self.p.set_val('seg_1:a', np.reshape(200 + np.arange(5), (5, 1)))

        self.p.set_val('seg_0:b', np.reshape(100 + np.arange(15), (5, 3)))
        self.p.set_val('seg_1:b', np.reshape(200 + np.arange(15), (5, 3)))

        self.p.set_val('seg_0:c', np.reshape(100 + np.arange(45), (5, 3, 3)))
        self.p.set_val('seg_1:c', np.reshape(200 + np.arange(45), (5, 3, 3)))

        self.p.run_model()

    def test_results(self):
        p = self.p
        assert_almost_equal(p['seg_0:a'],
                            p['path_constraints.seg_0_values:a'][0:5])

        assert_almost_equal(p['seg_1:a'],
                            p['path_constraints.seg_1_values:a'][0:5])

        assert_almost_equal(p['seg_0:b'],
                            p['path_constraints.seg_0_values:b'][0:5, ...])

        assert_almost_equal(p['seg_1:b'],
                            p['path_constraints.seg_1_values:b'][0:5, ...])

        assert_almost_equal(p['seg_0:c'],
                            p['path_constraints.seg_0_values:c'][0:5, ...])

        assert_almost_equal(p['seg_1:c'],
                            p['path_constraints.seg_1_values:c'][0:5, ...])

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
