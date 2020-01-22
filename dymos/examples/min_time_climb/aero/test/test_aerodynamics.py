import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om

from dymos.examples.min_time_climb.aero import AeroGroup


class TestAeroGroup(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem(model=om.Group())
        nn = 5

        ivc = om.IndepVarComp()

        ivc.add_output('rho', val=0.0001 * np.ones(nn), units='kg/m**3')
        ivc.add_output('v', val=0.0001 * np.ones(nn), units='m/s')
        ivc.add_output('S', val=0.0001 * np.ones(nn), units='m**2')
        ivc.add_output('alpha', val=np.zeros(nn), units='rad')
        ivc.add_output('sos', val=np.ones(nn), units='m/s')
        self.prob.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        self.prob.model.add_subsystem(name='aero', subsys=AeroGroup(num_nodes=nn))

        self.prob.model.connect('rho', 'aero.rho')
        self.prob.model.connect('v', 'aero.v')
        self.prob.model.connect('S', 'aero.S')
        self.prob.model.connect('alpha', 'aero.alpha')
        self.prob.model.connect('sos', 'aero.sos')

        self.prob.setup()

    def test_aero_values(self):

        self.prob['rho'] = 1.2250409
        self.prob['v'] = 115.824
        self.prob['S'] = 49.2386

        self.prob['alpha'] = np.radians(4.1900199)
        self.prob['sos'] = 340.29396
        self.prob.run_model()

        q_expected = 0.5 * self.prob['rho'] * self.prob['v']**2
        lift_expected = q_expected * self.prob['S'] * self.prob['aero.CL']   # [101779.451502]
        drag_expected = q_expected * self.prob['S'] * self.prob['aero.CD']   # [9278.85725577]

        assert_almost_equal(self.prob['aero.q'], q_expected, decimal=7)
        assert_almost_equal(self.prob['aero.f_lift'], lift_expected, decimal=7)
        assert_almost_equal(self.prob['aero.f_drag'], drag_expected, decimal=7)

    def testAeroDerivs(self):
        cpd = self.prob.check_partials(compact_print=True)

        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                np.testing.assert_almost_equal(actual=cpd[comp][var, wrt]['J_fwd'],
                                               desired=cpd[comp][var, wrt]['J_fd'],
                                               decimal=5,
                                               err_msg='Possible error in partials of component '
                                                       '{0} for {1} wrt {2}'.format(comp, var, wrt))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
