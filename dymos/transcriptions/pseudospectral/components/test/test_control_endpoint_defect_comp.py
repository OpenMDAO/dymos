import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.transcriptions.pseudospectral.components import ControlEndpointDefectComp
from dymos.transcriptions.grid_data import GridData


class TestControlEndpointDefectComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True
        gd = GridData(num_segments=2, segment_ends=np.array([0., 2., 4.]),
                      transcription='radau-ps', transcription_order=3)

        self.gd = gd

        self.p = om.Problem(model=om.Group())

        control_opts = {'u': {'units': 'm', 'shape': (1,), 'dynamic': True, 'opt': True},
                        'v': {'units': 'm', 'shape': (3, 2), 'dynamic': True, 'opt': True}}

        indep_comp = om.IndepVarComp()
        self.p.model.add_subsystem('indep', indep_comp, promotes=['*'])

        indep_comp.add_output('controls:u',
                              val=np.zeros((gd.subset_num_nodes['all'], 1)), units='m')

        indep_comp.add_output('controls:v',
                              val=np.zeros((gd.subset_num_nodes['all'], 3, 2)), units='m')

        self.p.model.add_subsystem('endpoint_defect_comp',
                                   subsys=ControlEndpointDefectComp(grid_data=gd,
                                                                    control_options=control_opts))

        self.p.model.connect('controls:u', 'endpoint_defect_comp.controls:u')
        self.p.model.connect('controls:v', 'endpoint_defect_comp.controls:v')

        self.p.setup(force_alloc_complex=True)

        self.p['controls:u'] = np.random.random((gd.subset_num_nodes['all'], 1))
        self.p['controls:v'] = np.random.random((gd.subset_num_nodes['all'], 3, 2))

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        u_coefs = np.polyfit(self.gd.node_ptau[-4:-1], self.p['controls:u'][-4:-1], deg=2)
        u_poly = np.poly1d(u_coefs.ravel())
        u_interp = u_poly(1.0)
        u_given = self.p['controls:u'][-1]
        assert_near_equal(np.ravel(self.p['endpoint_defect_comp.control_endpoint_defects:u']),
                          np.ravel(u_given - u_interp),
                          tolerance=1.0E-12)

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=False, method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
