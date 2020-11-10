import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om

import dymos as dm
from dymos.transcriptions.grid_data import GridData
from dymos.transcriptions.pseudospectral.components import CollocationComp
from dymos.transcriptions.pseudospectral.components.state_independents import StateIndependentsComp
from dymos.utils.testing_utils import assert_check_partials

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
CollocationComp = CompWrapperConfig(CollocationComp)
StateIndependentsComp = CompWrapperConfig(StateIndependentsComp)


class TestCollocationCompSolOpt(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def make_prob(self, transcription, n_segs, order, compressed):

        p = om.Problem(model=om.Group())

        gd = GridData(num_segments=n_segs, segment_ends=np.arange(n_segs+1),
                      transcription=transcription, transcription_order=order, compressed=compressed)

        state_options = {'x': {'units': 'm', 'shape': (1, ), 'fix_initial': True,
                               'fix_final': False, 'solve_segments': False,
                               'connected_initial': False},
                         'v': {'units': 'm/s', 'shape': (3, 2), 'fix_initial': False,
                               'fix_final': True, 'solve_segments': True,
                               'connected_initial': False}}

        indep_comp = om.IndepVarComp()
        p.model.add_subsystem('indep', indep_comp, promotes_outputs=['*'])

        indep_comp.add_output(
            'dt_dstau',
            val=np.zeros((gd.subset_num_nodes['col']))
        )

        indep_comp.add_output(
            'f_approx:x',
            val=np.zeros((gd.subset_num_nodes['col'], 1)), units='m')
        indep_comp.add_output(
            'f_computed:x',
            val=np.zeros((gd.subset_num_nodes['col'], 1)), units='m')

        indep_comp.add_output(
            'f_approx:v',
            val=np.zeros((gd.subset_num_nodes['col'], 3, 2)), units='m/s')
        indep_comp.add_output(
            'f_computed:v',
            val=np.zeros((gd.subset_num_nodes['col'], 3, 2)), units='m/s')

        p.model.add_subsystem('defect_comp',
                              subsys=CollocationComp(grid_data=gd, state_options=state_options))

        indep = StateIndependentsComp(grid_data=gd, state_options=state_options)
        p.model.add_subsystem('state_indep', indep, promotes_outputs=['*'])

        p.model.connect('f_approx:x', 'defect_comp.f_approx:x')
        p.model.connect('f_approx:v', 'defect_comp.f_approx:v')
        p.model.connect('f_computed:x', 'defect_comp.f_computed:x')
        p.model.connect('f_computed:v', 'defect_comp.f_computed:v')
        p.model.connect('dt_dstau', 'defect_comp.dt_dstau')
        p.model.connect('defect_comp.defects:v', 'state_indep.defects:v')

        p.setup(force_alloc_complex=True)

        p['dt_dstau'] = np.random.random(gd.subset_num_nodes['col'])

        p['f_approx:x'] = np.random.random((gd.subset_num_nodes['col'], 1))
        p['f_approx:v'] = np.random.random((gd.subset_num_nodes['col'], 3, 2))

        p['f_computed:x'] = np.random.random((gd.subset_num_nodes['col'], 1))
        p['f_computed:v'] = np.random.random((gd.subset_num_nodes['col'], 3, 2))

        p.run_model()
        p.model.run_apply_nonlinear()

        # p.model.list_outputs(residuals=True, print_arrays=True)

        return p

    def test_results(self):
        p = self.make_prob('gauss-lobatto', n_segs=4, order=3, compressed=False)

        dt_dstau = p['dt_dstau']

        assert_almost_equal(p['defect_comp.defects:x'],
                            dt_dstau[:, np.newaxis] * (p['f_approx:x']-p['f_computed:x']))

        solver_nodes = p.model.state_indep.solver_node_idx[:-1]  # fix_final
        assert_almost_equal(p.model._residuals['state_indep.states:v'][solver_nodes],
                            dt_dstau[:, np.newaxis, np.newaxis] *
                            (p['f_approx:v']-p['f_computed:v']))

        p = self.make_prob('gauss-lobatto', n_segs=4, order=3, compressed=True)

        dt_dstau = p['dt_dstau']

        assert_almost_equal(p['defect_comp.defects:x'],
                            dt_dstau[:, np.newaxis] * (p['f_approx:x']-p['f_computed:x']))

        solver_nodes = p.model.state_indep.solver_node_idx[:-1]  # fix_final
        assert_almost_equal(p.model._residuals['state_indep.states:v'][solver_nodes],
                            dt_dstau[:, np.newaxis, np.newaxis] *
                            (p['f_approx:v']-p['f_computed:v']))

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1e1000)

        p = self.make_prob('radau-ps', n_segs=2, order=5, compressed=False)
        cpd = p.check_partials(compact_print=True, method='fd')
        del cpd['state_indep']
        assert_check_partials(cpd)

        p = self.make_prob('radau-ps', n_segs=2, order=5, compressed=True)
        cpd = p.check_partials(compact_print=True, method='fd')
        del cpd['state_indep']
        assert_check_partials(cpd)

        p = self.make_prob('gauss-lobatto', n_segs=3, order=5, compressed=False)
        cpd = p.check_partials(compact_print=True, method='fd')
        del cpd['state_indep']
        assert_check_partials(cpd)

        p = self.make_prob('gauss-lobatto', n_segs=4, order=3, compressed=True)
        cpd = p.check_partials(compact_print=True, method='fd')
        del cpd['state_indep']
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
