import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs

from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE


@use_tempdirs
class TestFiniteBurnEOM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 2

        p = cls.p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        p.model.add_subsystem('ode', subsys=FiniteBurnODE(num_nodes=nn), promotes_inputs=['*'],
                              promotes_outputs=['*'])

        ivc.add_output('r',
                       val=np.ones(nn),
                       desc='radius from center of attraction',
                       units='DU')

        ivc.add_output('theta',
                       val=np.zeros(nn),
                       desc='anomaly term',
                       units='rad')

        ivc.add_output('vr',
                       val=np.zeros(nn),
                       desc='local vertical velocity component',
                       units='DU/TU')

        ivc.add_output('vt',
                       val=np.zeros(nn),
                       desc='local horizontal velocity component',
                       units='DU/TU')

        ivc.add_output('accel',
                       val=np.zeros(nn),
                       desc='acceleration due to thrust',
                       units='DU/TU**2')

        ivc.add_output('u1',
                       val=np.zeros(nn),
                       desc='thrust angle above local horizontal',
                       units='rad')

        ivc.add_output('c',
                       val=np.zeros(nn),
                       desc='exhaust velocity',
                       units='DU/TU')

        p.setup()

        p['r'] = 2.0
        p['theta'] = 0.05
        p['vr'] = 0.2
        p['vt'] = 1.0
        p['accel'] = [0.1, 0.1]
        p['u1'] = [0.0, np.pi]
        p['c'] = 9.80665 * 2000

        p.run_model()

    def test_outputs(self):
        p = self.p
        assert_near_equal(p['r_dot'], p['vr'])
        assert_near_equal(p['theta_dot'], p['vt'] / p['r'])
        assert_near_equal(p['at_dot'], p['accel']**2 / p['c'])

    def test_partials(self):
        cpd = self.p.check_partials(compact_print=False)
        assert_check_partials(cpd, atol=1.0E-5, rtol=2.0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
