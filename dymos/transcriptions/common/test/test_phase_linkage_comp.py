import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from dymos.transcriptions.common import PhaseLinkageComp


class TestPhaseLinkageComp(unittest.TestCase):

    def setUp(self):

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

        linkage_comp.add_linkage('L01a', vars=('x',), equals=0.0, shape=(1,), units='m')
        linkage_comp.add_linkage('L01b', vars=('u',), equals=0.0, shape=(3,), units='deg')
        linkage_comp.add_linkage('L01c', vars=('v',), equals=0.0, shape=(3, 3), units='N')

        self.p.model.add_subsystem('linkage_comp', subsys=linkage_comp)

        self.p.model.connect('phase0:x', 'linkage_comp.L01a_x:lhs',
                             src_indices=[-1],
                             flat_src_indices=True)

        self.p.model.connect('phase1:x', 'linkage_comp.L01a_x:rhs',
                             src_indices=[0],
                             flat_src_indices=True)

        self.p.model.connect('phase0:u', 'linkage_comp.L01b_u:lhs',
                             src_indices=np.arange(-3, 0, dtype=int),
                             flat_src_indices=True)

        self.p.model.connect('phase1:u', 'linkage_comp.L01b_u:rhs',
                             src_indices=np.arange(0, 3, dtype=int),
                             flat_src_indices=True)

        self.p.model.connect('phase0:v', 'linkage_comp.L01c_v:lhs',
                             src_indices=np.arange(-9, 0, dtype=int).reshape((3, 3)),
                             flat_src_indices=True)

        self.p.model.connect('phase1:v', 'linkage_comp.L01c_v:rhs',
                             src_indices=np.arange(0, 9, dtype=int).reshape((3, 3)),
                             flat_src_indices=True)

        self.p.setup()

        self.p['phase0:x'] = np.random.rand(*self.p['phase0:x'].shape)
        self.p['phase0:u'] = np.random.rand(*self.p['phase0:u'].shape)
        self.p['phase0:v'] = np.random.rand(*self.p['phase0:v'].shape)

        self.p['phase1:x'] = np.random.rand(*self.p['phase1:x'].shape)
        self.p['phase1:u'] = np.random.rand(*self.p['phase1:u'].shape)
        self.p['phase1:v'] = np.random.rand(*self.p['phase1:v'].shape)

        self.p.run_model()

    def test_results(self):

        assert_almost_equal(self.p['phase1:x'][0, ...] - self.p['phase0:x'][-1, ...],
                            self.p['linkage_comp.L01a_x'])

        assert_almost_equal(self.p['phase1:u'][0, ...] - self.p['phase0:u'][-1, ...],
                            self.p['linkage_comp.L01b_u'])

        assert_almost_equal(self.p['phase1:v'][0, ...] - self.p['phase0:v'][-1, ...],
                            self.p['linkage_comp.L01c_v'])

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
