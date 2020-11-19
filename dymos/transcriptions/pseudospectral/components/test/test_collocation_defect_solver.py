import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.transcriptions.pseudospectral.components.collocation_comp import CollocationComp
from dymos.transcriptions.pseudospectral.components.state_independents import StateIndependentsComp

# Modify class so we can run it standalone.
import dymos as dm
from dymos.utils.misc import CompWrapperConfig
CollocationComp = CompWrapperConfig(CollocationComp)
StateIndependentsComp = CompWrapperConfig(StateIndependentsComp)


class TestCollocationBalanceIndex(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def make_prob(self, transcription, n_segs, order, compressed):

        p = om.Problem(model=om.Group())

        gd = GridData(num_segments=n_segs, segment_ends=np.arange(n_segs+1),
                      transcription=transcription, transcription_order=order, compressed=compressed)

        state_options = {'x': {'units': 'm', 'shape': (1, ), 'fix_initial': True,
                               'fix_final': False, 'solve_segments': True,
                               'connected_initial': False, 'connected_final': False},
                         'v': {'units': 'm/s', 'shape': (3, 2), 'fix_initial': False,
                               'fix_final': True, 'solve_segments': True,
                               'connected_initial': False, 'connected_final': False}}

        subsys = StateIndependentsComp(grid_data=gd, state_options=state_options)
        p.model.add_subsystem('defect_comp', subsys=subsys)

        p.setup()
        p.final_setup()

        return p

    def test_3_lgl(self):

        p = self.make_prob(transcription='gauss-lobatto', n_segs=3, order=3, compressed=False)
        defect_comp = p.model.defect_comp
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1, 3, 5]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 2, 4]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0, 1, 3]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([2, 4, 5]))

    def test_5_lgl(self):

        p = self.make_prob(transcription='gauss-lobatto', n_segs=2, order=5, compressed=False)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1, 2, 4, 5]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 3]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0, 1, 2, 4]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([3, 5]))

    def test_3_lgr(self):

        p = self.make_prob(transcription='radau-ps', n_segs=3, order=3, compressed=False)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']),
                            set([1, 2, 3, 5, 6, 7, 9, 10, 11]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 4, 8]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']),
                            set([0, 1, 2, 3, 5, 6, 7, 9, 10]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([4, 8, 11]))

    def test_5_lgr(self):

        p = self.make_prob(transcription='radau-ps', n_segs=2, order=5, compressed=False)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']),
                            set([1, 2, 3, 4, 5, 7, 8, 9, 10, 11]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, 6]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']),
                            set([0, 1, 2, 3, 4, 5, 7, 8, 9, 10]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([6, 11]))

    def test_5_lgr_compressed(self):

        p = self.make_prob(transcription='radau-ps', n_segs=2, order=5, compressed=True)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']),
                            set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, ]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']),
                            set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([10, ]))

    def test_5_lgl_compressed(self):

        p = self.make_prob(transcription='gauss-lobatto', n_segs=2, order=5, compressed=True)
        defect_comp = p.model.defect_comp

        self.assertSetEqual(set(defect_comp.state_idx_map['x']['solver']), set([1, 2, 3, 4]))
        self.assertSetEqual(set(defect_comp.state_idx_map['x']['indep']), set([0, ]))

        self.assertSetEqual(set(defect_comp.state_idx_map['v']['solver']), set([0, 1, 2, 3]))
        self.assertSetEqual(set(defect_comp.state_idx_map['v']['indep']), set([4, ]))


class TestCollocationBalanceApplyNL(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def make_prob(self, transcription, n_segs, order, compressed):

        gd = GridData(
            num_segments=n_segs, segment_ends=np.arange(n_segs+1),
            transcription=transcription, transcription_order=order, compressed=compressed)

        state_options = {'x': {'units': 'm', 'shape': (1, ), 'fix_initial': True,
                               'fix_final': False, 'solve_segments': True,
                               'connected_initial': False},
                         'v': {'units': 'm/s', 'shape': (3, 2), 'fix_initial': True,
                               'fix_final': False, 'solve_segments': True,
                               'connected_initial': False}}

        num_col_nodes = gd.subset_num_nodes['col']
        num_col_nodes_per_seg = gd.subset_num_nodes_per_segment['col']

        p = om.Problem(model=om.Group())

        indep_comp = om.IndepVarComp()
        p.model.add_subsystem('indep', indep_comp, promotes_outputs=['*'])

        indep_comp.add_output(
            'dt_dstau',
            val=np.repeat(np.arange(n_segs)+1, num_col_nodes_per_seg)
        )

        indep_comp.add_output(
            'f_approx:x',
            val=np.ones((num_col_nodes, 1)), units='m')
        indep_comp.add_output(
            'f_computed:x',
            val=np.ones((num_col_nodes, 1))*2, units='m')

        indep_comp.add_output(
            'f_approx:v',
            val=np.ones((num_col_nodes, 3, 2)), units='m/s')
        indep_comp.add_output(
            'f_computed:v',
            val=np.ones((num_col_nodes, 3, 2))*2, units='m/s')

        p.model.add_subsystem('defect_comp',
                              subsys=CollocationComp(grid_data=gd,
                                                     state_options=state_options))

        indep = StateIndependentsComp(grid_data=gd, state_options=state_options)
        p.model.add_subsystem('state_indep', indep, promotes_outputs=['*'])

        p.model.connect('f_approx:x', 'defect_comp.f_approx:x')
        p.model.connect('f_approx:v', 'defect_comp.f_approx:v')
        p.model.connect('f_computed:x', 'defect_comp.f_computed:x')
        p.model.connect('f_computed:v', 'defect_comp.f_computed:v')
        p.model.connect('dt_dstau', 'defect_comp.dt_dstau')
        p.model.connect('defect_comp.defects:x', 'state_indep.defects:x')
        p.model.connect('defect_comp.defects:v', 'state_indep.defects:v')

        p.setup(force_alloc_complex=True)

        return p

    def test_apply_nonlinear(self):

        p = self.make_prob('gauss-lobatto', n_segs=3, order=3, compressed=False)

        p.run_model()
        p.model.run_apply_nonlinear()  # need to make sure residuals are computed

        expected = np.array([0., -1., 0., -2., 0., -3.])

        assert_almost_equal(p.model._residuals._views['state_indep.states:x'],
                            expected.reshape(6, 1))

        assert_almost_equal(p.model._residuals._views['state_indep.states:v'],
                            expected[:, np.newaxis, np.newaxis]*np.ones((6, 3, 2)))

    def test_partials(self):

        def assert_partials(data):
            # assert_check_partials(cpd) # can't use this here, cause of indepvarcomp weirdness
            for of, wrt in data:
                if of == wrt:
                    # IndepVarComp like outputs have correct derivs, but FD is wrong so we skip
                    # them (should be some form of -I)
                    continue
                check_data = data[(of, wrt)]
                self.assertLess(check_data['abs error'].forward, 1e-8)

            # print((self.p['f_approx:v']-self.p['f_computed:v']).ravel())

        np.set_printoptions(linewidth=1024, edgeitems=1e1000)

        p = self.make_prob('radau-ps', n_segs=2, order=5, compressed=False)
        cpd = p.check_partials(compact_print=True, method='fd')
        data = cpd['defect_comp']
        assert_partials(data)

        p = self.make_prob('radau-ps', n_segs=2, order=5, compressed=True)
        cpd = p.check_partials(compact_print=True, method='fd')
        data = cpd['defect_comp']
        assert_partials(data)

        p = self.make_prob('gauss-lobatto', n_segs=3, order=5, compressed=False)
        cpd = p.check_partials(compact_print=True, method='fd')
        data = cpd['defect_comp']
        assert_partials(data)

        p = self.make_prob('gauss-lobatto', n_segs=4, order=3, compressed=True)
        cpd = p.check_partials(compact_print=True, method='fd')
        data = cpd['defect_comp']
        assert_partials(data)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
