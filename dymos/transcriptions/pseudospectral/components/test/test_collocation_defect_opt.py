import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.pseudospectral.components import CollocationComp
from dymos.transcriptions.grid_data import GridData

# Modify class so we can run it standalone.
from dymos.utils.misc import CompWrapperConfig
CollocationComp = CompWrapperConfig(CollocationComp)


class TestCollocationComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True
        transcription = 'gauss-lobatto'

        gd = GridData(
            num_segments=4, segment_ends=np.array([0., 2., 4., 5., 12.]),
            transcription=transcription, transcription_order=3)

        self.p = om.Problem(model=om.Group())

        state_options = {'x': {'units': 'm', 'shape': (1,), 'fix_initial': True,
                               'fix_final': False, 'solve_segments': False,
                               'connected_initial': False, 'connected_final': False},
                         'v': {'units': 'm/s', 'shape': (3, 2), 'fix_initial': False,
                               'fix_final': True, 'solve_segments': False,
                               'connected_initial': False, 'connected_final': False}}

        indep_comp = om.IndepVarComp()
        self.p.model.add_subsystem('indep', indep_comp, promotes_outputs=['*'])

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

        self.p.model.add_subsystem('defect_comp',
                                   subsys=CollocationComp(grid_data=gd,
                                                          state_options=state_options))

        self.p.model.connect('f_approx:x', 'defect_comp.f_approx:x')
        self.p.model.connect('f_approx:v', 'defect_comp.f_approx:v')
        self.p.model.connect('f_computed:x', 'defect_comp.f_computed:x')
        self.p.model.connect('f_computed:v', 'defect_comp.f_computed:v')
        self.p.model.connect('dt_dstau', 'defect_comp.dt_dstau')

        self.p.setup(force_alloc_complex=True)

        self.p['dt_dstau'] = np.random.random(gd.subset_num_nodes['col'])

        self.p['f_approx:x'] = np.random.random((gd.subset_num_nodes['col'], 1))
        self.p['f_approx:v'] = np.random.random((gd.subset_num_nodes['col'], 3, 2))

        self.p['f_computed:x'] = np.random.random((gd.subset_num_nodes['col'], 1))
        self.p['f_computed:v'] = np.random.random((gd.subset_num_nodes['col'], 3, 2))

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):
        dt_dstau = self.p['dt_dstau']

        assert_almost_equal(self.p['defect_comp.defects:x'],
                            dt_dstau[:, np.newaxis] * (self.p['f_approx:x']-self.p['f_computed:x']))

        assert_almost_equal(self.p['defect_comp.defects:v'],
                            dt_dstau[:, np.newaxis, np.newaxis] *
                            (self.p['f_approx:v']-self.p['f_computed:v']))

    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(compact_print=False, method='fd', out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
