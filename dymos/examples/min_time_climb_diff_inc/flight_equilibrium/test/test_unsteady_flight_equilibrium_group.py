from __future__ import print_function, division, absolute_import

import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp

from dymos.examples.min_time_climb_diff_inc.flight_equilibrium.unsteady_flight_equilibrium_group \
    import UnsteadyFlightEquilibriumGroup


class TestAeroGroup(unittest.TestCase):

    def setUp(self):
        self.prob = Problem(model=Group())
        nn = 10

        ivc = IndepVarComp()

        ivc.add_output('m', val=19030.468 * np.ones(nn), units='kg')
        ivc.add_output('h', val=10000.0 * np.ones(nn), units='m')
        ivc.add_output('rho', val=1.225 * np.ones(nn), units='kg/m**3')
        ivc.add_output('v', val=115.824 * np.ones(nn), units='m/s')
        ivc.add_output('S', val=49.2386 * np.ones(nn), units='m**2')
        ivc.add_output('sos', val=340.29396*np.ones(nn), units='m/s')
        ivc.add_output('Isp', val=1600*np.ones(nn), units='s')
        ivc.add_output('gam', val=0.0*np.ones(nn), units='deg')
        ivc.add_output('gam_dot_approx', val=0.0*np.ones(nn), units='deg/s')
        ivc.add_output('v_dot_approx', val=0.0*np.ones(nn), units='m/s**2')
        self.prob.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        self.prob.model.add_subsystem(name='feg',
                                      subsys=UnsteadyFlightEquilibriumGroup(num_nodes=nn))

        self.prob.model.connect('h', 'feg.prop.h')
        self.prob.model.connect('rho', 'feg.aero.rho')
        self.prob.model.connect('m', ('feg.flight_dynamics.m'))
        self.prob.model.connect('v', ('feg.aero.v', 'feg.flight_dynamics.v'))
        self.prob.model.connect('S', 'feg.aero.S')
        self.prob.model.connect('sos', 'feg.aero.sos')
        self.prob.model.connect('Isp', 'feg.prop.Isp')
        self.prob.model.connect('gam', 'feg.flight_dynamics.gam')
        self.prob.model.connect('gam_dot_approx', 'feg.balance_comp.gam_dot_approx')
        self.prob.model.connect('v_dot_approx', 'feg.balance_comp.v_dot_approx')

        self.prob.setup()

    def test_aero_values(self):

        self.prob['rho'] = 1.225
        self.prob['v'] = 115.824
        self.prob['v'] = 500
        self.prob['v'] = np.linspace(100, 500, 10)
        self.prob['S'] = 49.2386

        self.prob['sos'] = 340.29396
        self.prob['feg.balance_comp.throttle'] = 0.75
        self.prob['feg.balance_comp.alpha'] = 0.67
        self.prob['v_dot_approx'] = 0.0
        self.prob['gam_dot_approx'] = 0.0

        self.prob.run_model()

        self.prob.model.list_outputs(print_arrays=True, residuals=True)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
