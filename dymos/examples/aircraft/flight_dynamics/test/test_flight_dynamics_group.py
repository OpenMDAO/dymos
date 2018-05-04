from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.examples.aircraft.flight_dynamics.flight_dynamics_group import FlightDynamicsGroup


class TestFlightDynamicsGroup(unittest.TestCase):

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
        ivc.add_output('m', val=np.ones(cls.n), units='kg', desc='aircraft dry mass')
        ivc.add_output('q', val=0.5 * 200.0**2 * np.ones(cls.n), units='Pa', desc='dynamic pressure')

        cls.p.model.add_subsystem('flight_dynamics', FlightDynamicsGroup(num_nodes=cls.n))

        cls.p.model.connect('alt', 'flight_dynamics.alt')
        cls.p.model.connect('TAS', 'flight_dynamics.TAS')
        cls.p.model.connect('TAS_rate', 'flight_dynamics.TAS_rate')
        cls.p.model.connect('gam', 'flight_dynamics.gam')
        cls.p.model.connect('gam_rate', 'flight_dynamics.gam_rate')
        cls.p.model.connect('S', 'flight_dynamics.aero.S')
        cls.p.model.connect('mach', 'flight_dynamics.mach')
        cls.p.model.connect('m', 'flight_dynamics.m')
        cls.p.model.connect('q', 'flight_dynamics.q')

        cls.p.setup(check=True, mode='fwd')

        cls.p.run_model()

    def test_results(self):
        print()
        print('flight_dynamics.alpha', self.p['flight_dynamics.alpha'])
        print('flight_dynamics.thrust', self.p['flight_dynamics.thrust'])
        print('flight_dynamics.eta', self.p['flight_dynamics.eta'])
        print('flight_dynamics.aero.CM', self.p['flight_dynamics.aero.CM'])
        print('flight_dynamics.aero.D', self.p['flight_dynamics.aero.D'])
        print('flight_dynamics.aero.L', self.p['flight_dynamics.aero.L'])
        print('aircraft weight', self.p['m']*9.80665)

    # def test_partials(self):
    #     cpd = self.p.check_partials(suppress_output=True)
    #     assert_check_partials(cpd)
