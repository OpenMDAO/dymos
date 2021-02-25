import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.pseudospectral.components import GaussLobattoInterleaveComp
from dymos.transcriptions.grid_data import GridData


class TestGaussLobattoInterleaveComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

        self.grid_data = gd = GridData(num_segments=3, segment_ends=np.array([0., 2., 4., 10.0]),
                                       transcription='gauss-lobatto', transcription_order=[3, 3, 3])

        num_disc_nodes = gd.subset_num_nodes['state_disc']
        num_col_nodes = gd.subset_num_nodes['col']

        self.p = om.Problem(model=om.Group())

        state_options = {'u': {'units': 'm', 'shape': (1,)},
                         'v': {'units': 'm', 'shape': (3, 2)}}

        ode_outputs = {'vehicle_cg': {'units': 'm', 'shape': (3,)}}

        indep_comp = om.IndepVarComp()
        self.p.model.add_subsystem('indep', indep_comp, promotes=['*'])

        indep_comp.add_output('state_disc:u',
                              val=np.zeros((num_disc_nodes, 1)), units='m')

        indep_comp.add_output('state_disc:v',
                              val=np.zeros((num_disc_nodes, 3, 2)), units='m')

        indep_comp.add_output('state_col:u',
                              val=np.zeros((num_col_nodes, 1)), units='m')

        indep_comp.add_output('state_col:v',
                              val=np.zeros((num_col_nodes, 3, 2)), units='m')

        indep_comp.add_output('ode_disc:cg',
                              val=np.zeros((num_disc_nodes, 3)), units='m')

        indep_comp.add_output('ode_col:cg',
                              val=np.zeros((num_col_nodes, 3)), units='m')

        glic = self.p.model.add_subsystem('interleave_comp',
                                          subsys=GaussLobattoInterleaveComp(grid_data=gd))

        glic.add_var('u', **state_options['u'], disc_src='state_disc:u', col_src='state_col:u')
        glic.add_var('v', **state_options['v'], disc_src='state_disc:v', col_src='state_col:v')
        glic.add_var('vehicle_cg', **ode_outputs['vehicle_cg'], disc_src='ode_disc:cg', col_src='ode_col:cg')

        self.p.model.connect('state_disc:u', 'interleave_comp.disc_values:u')
        self.p.model.connect('state_disc:v', 'interleave_comp.disc_values:v')
        self.p.model.connect('state_col:u', 'interleave_comp.col_values:u')
        self.p.model.connect('state_col:v', 'interleave_comp.col_values:v')

        self.p.model.connect('ode_disc:cg', 'interleave_comp.disc_values:vehicle_cg')
        self.p.model.connect('ode_col:cg', 'interleave_comp.col_values:vehicle_cg')

        self.p.setup(force_alloc_complex=True)

        self.p['state_disc:u'] = np.random.random((num_disc_nodes, 1))
        self.p['state_disc:v'] = np.random.random((num_disc_nodes, 3, 2))
        self.p['state_col:u'] = np.random.random((num_col_nodes, 1))
        self.p['state_col:v'] = np.random.random((num_col_nodes, 3, 2))

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        u_disc = self.p.get_val('state_disc:u')
        v_disc = self.p.get_val('state_disc:v')
        u_col = self.p.get_val('state_col:u')
        v_col = self.p.get_val('state_col:v')

        u_all = self.p.get_val('interleave_comp.all_values:u')
        v_all = self.p.get_val('interleave_comp.all_values:v')

        assert_near_equal(u_all[self.grid_data.subset_node_indices['state_disc'], ...],
                          u_disc)

        assert_near_equal(v_all[self.grid_data.subset_node_indices['state_disc'], ...],
                          v_disc)

        assert_near_equal(u_all[self.grid_data.subset_node_indices['col'], ...],
                          u_col)

        assert_near_equal(v_all[self.grid_data.subset_node_indices['col'], ...],
                          v_col)

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=True, method='cs', out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
