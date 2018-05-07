from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.examples.aircraft.flight_equlibrium.flight_equilibrium_group import FlightEquilibriumGroup


class TestFlightEquilibriumGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 10

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('alt', val=3000 * np.ones(cls.n), units='m', desc='altitude above MSL')
        ivc.add_output('TAS', val=200.0 * np.ones(cls.n), units='m/s', desc='true airspeed')
        ivc.add_output('TAS_rate', val=np.zeros(cls.n), units='m/s**2', desc='acceleration')
        ivc.add_output('gam', val=np.zeros(cls.n), units='rad', desc='flight path angle')
        ivc.add_output('gam_rate', val=np.zeros(cls.n), units='rad/s', desc='flight path angle rate')
        ivc.add_output('S', val= 427.8 * np.ones(cls.n), units='m**2', desc='reference area')
        ivc.add_output('mach', val= 0.5 * np.ones(cls.n), units=None, desc='mach number')
        ivc.add_output('mass', val=200000 * np.ones(cls.n), units='kg', desc='aircraft dry mass')
        ivc.add_output('q', val=0.5 * 200.0**2 * np.ones(cls.n), units='Pa', desc='dynamic pressure')

        cls.p.model.add_subsystem('flight_equilibrium', FlightEquilibriumGroup(num_nodes=cls.n),
                                  promotes_inputs=['aero.*', 'flight_dynamics.*'],
                                  promotes_outputs=['aero.*', 'flight_dynamics.*'])

        cls.p.model.connect('alt', ('aero.alt'))
        cls.p.model.connect('TAS', 'flight_dynamics.TAS')
        cls.p.model.connect('TAS_rate', 'flight_equilibrium.TAS_rate')
        cls.p.model.connect('gam', 'flight_dynamics.gam')
        cls.p.model.connect('gam_rate', 'flight_equilibrium.gam_rate')
        cls.p.model.connect('S', 'aero.S')
        cls.p.model.connect('mach', 'aero.mach')
        cls.p.model.connect('mass', 'flight_dynamics.mass')
        cls.p.model.connect('q', 'aero.q')

        cls.p.setup(check=True, mode='fwd')

        cls.p.run_model()

    def test_results(self):

        alpha = self.p['flight_dynamics.alpha']
        L = self.p['aero.L']
        D = self.p['aero.D']
        T = self.p['flight_dynamics.thrust']
        mass = self.p['mass']
        weight = mass * 9.80665

        assert_rel_error(self,  (T * np.sin(alpha) + L), weight)
        assert_rel_error(self, T * np.cos(alpha), D)

    @unittest.skip('skip for now')
    def test_partials(self):
        np.set_printoptions(linewidth=1024)
        cpd = self.p.check_partials(suppress_output=False, method='fd', step=1.0E-6)
        # assert_check_partials(cpd, atol=1.0E-6, rtol=2.0)

