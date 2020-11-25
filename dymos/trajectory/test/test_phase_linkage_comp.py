import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

import dymos as dm
from dymos.trajectory.phase_linkage_comp import PhaseLinkageComp
from dymos.trajectory.options import LinkageOptionsDictionary


class TestPhaseLinkageComp(unittest.TestCase):

    def setUp(self):
        dm.options['include_check_partials'] = True
        self.p = om.Problem(model=om.Group())

        ivp = self.p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        ndn = 20
        nn = 30

        ivp.add_output('phase0:x', val=np.zeros((ndn, 1)), units='m')
        ivp.add_output('phase0:u', val=np.zeros((nn, 3)), units='deg')
        ivp.add_output('phase0:v', val=np.zeros((nn, 3, 3)), units='N')

        ivp.add_output('phase1:x', val=np.zeros((ndn, 1)), units='m')
        ivp.add_output('phase1:u', val=np.zeros((nn, 3)), units='deg')
        ivp.add_output('phase1:v', val=np.zeros((nn, 3, 3)), units='N')

        linkage_comp = PhaseLinkageComp()

        x_lnk = LinkageOptionsDictionary()
        x_lnk['phase_a'] = 'phase0'
        x_lnk['phase_b'] = 'phase1'
        x_lnk['var_a'] = 'x'
        x_lnk['var_b'] = 'x'
        x_lnk['loc_a'] = 'final'
        x_lnk['loc_b'] = 'initial'
        x_lnk['units'] = 'm'
        x_lnk['shape'] = (1,)
        x_lnk['equals'] = 0.0

        u_lnk = LinkageOptionsDictionary()
        u_lnk['phase_a'] = 'phase0'
        u_lnk['phase_b'] = 'phase1'
        u_lnk['var_a'] = 'u'
        u_lnk['var_b'] = 'u'
        u_lnk['loc_a'] = 'final'
        u_lnk['loc_b'] = 'initial'
        u_lnk['units'] = 'deg'
        u_lnk['shape'] = (3,)
        u_lnk['equals'] = 0.0

        v_lnk = LinkageOptionsDictionary()
        v_lnk['phase_a'] = 'phase0'
        v_lnk['phase_b'] = 'phase1'
        v_lnk['var_a'] = 'v'
        v_lnk['var_b'] = 'v'
        v_lnk['loc_a'] = 'final'
        v_lnk['loc_b'] = 'initial'
        v_lnk['units'] = 'N'
        v_lnk['shape'] = (3, 3)
        v_lnk['equals'] = 0.0

        linkage_comp.add_linkage_configure(x_lnk)
        linkage_comp.add_linkage_configure(u_lnk)
        linkage_comp.add_linkage_configure(v_lnk)

        self.p.model.add_subsystem('linkage_comp', subsys=linkage_comp)

        self.p.model.connect('phase0:x', 'linkage_comp.phase0:x',
                             src_indices=om.slicer[[0, -1], ...])

        self.p.model.connect('phase1:x', 'linkage_comp.phase1:x',
                             src_indices=om.slicer[[0, -1], ...])

        self.p.model.connect('phase0:u', 'linkage_comp.phase0:u',
                             src_indices=om.slicer[[0, -1], ...])

        self.p.model.connect('phase1:u', 'linkage_comp.phase1:u',
                             src_indices=om.slicer[[0, -1], ...])

        self.p.model.connect('phase0:v', 'linkage_comp.phase0:v',
                             src_indices=om.slicer[[0, -1], ...])

        self.p.model.connect('phase1:v', 'linkage_comp.phase1:v',
                             src_indices=om.slicer[[0, -1], ...])

        self.p.setup()

        self.p['phase0:x'] = np.random.rand(*self.p['phase0:x'].shape)
        self.p['phase0:u'] = np.random.rand(*self.p['phase0:u'].shape)
        self.p['phase0:v'] = np.random.rand(*self.p['phase0:v'].shape)

        self.p['phase1:x'] = np.random.rand(*self.p['phase1:x'].shape)
        self.p['phase1:u'] = np.random.rand(*self.p['phase1:u'].shape)
        self.p['phase1:v'] = np.random.rand(*self.p['phase1:v'].shape)

        self.p.run_model()

    def tearDown(self):
        dm.options['include_check_partials'] = False

    def test_results(self):

        assert_almost_equal(-self.p['phase1:x'][0, ...] + self.p['phase0:x'][-1, ...],
                            self.p['linkage_comp.phase0:x_final|phase1:x_initial'])

        assert_almost_equal(-self.p['phase1:u'][0, ...] + self.p['phase0:u'][-1, ...],
                            self.p['linkage_comp.phase0:u_final|phase1:u_initial'])

        assert_almost_equal(-self.p['phase1:v'][0, ...] + self.p['phase0:v'][-1, ...],
                            self.p['linkage_comp.phase0:v_final|phase1:v_initial'])

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
