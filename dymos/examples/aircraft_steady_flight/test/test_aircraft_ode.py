from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp, CSCJacobian, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE


class TestAircraftODEGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 10

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('mass_fuel', val=20000 * np.ones(cls.n), units='kg',
                       desc='aircraft fuel mass')
        ivc.add_output('mass_payload', val=84.02869 * 400 * np.ones(cls.n), units='kg',
                       desc='aircraft fuel mass')
        ivc.add_output('mass_empty', val=0.15E6 * np.ones(cls.n), units='kg',
                       desc='aircraft empty mass')
        ivc.add_output('alt', val=9.144*np.ones(cls.n), units='km', desc='altitude')
        ivc.add_output('climb_rate', val=np.zeros(cls.n), units='m/s', desc='climb rate')
        ivc.add_output('TAS', val=250.0*np.ones(cls.n), units='m/s', desc='true airspeed')
        ivc.add_output('S', val=427.8 * np.ones(cls.n), units='m**2', desc='reference area')

        cls.p.model.add_subsystem('ode', AircraftODE(num_nodes=cls.n))

        cls.p.model.connect('mass_fuel', 'ode.mass_comp.mass_fuel')
        cls.p.model.connect('mass_payload', 'ode.mass_comp.mass_payload')
        cls.p.model.connect('mass_empty', 'ode.mass_comp.mass_empty')
        cls.p.model.connect('alt', ['ode.atmos.h', 'ode.propulsion.alt', 'ode.aero.alt'])
        cls.p.model.connect('TAS', ['ode.mach_comp.TAS', 'ode.gam_comp.TAS', 'ode.q_comp.TAS',
                                    'ode.range_rate_comp.TAS'])
        cls.p.model.connect('climb_rate', ['ode.gam_comp.climb_rate'])
        cls.p.model.connect('S', ('ode.aero.S', 'ode.flight_equilibrium.S', 'ode.propulsion.S'))

        cls.p.model.jacobian = CSCJacobian()
        cls.p.model.linear_solver = DirectSolver()

        cls.p.setup(check=True, mode='fwd')

        cls.p.run_model()

    def test_results(self):
        print('dXdt:mass_fuel', self.p['ode.propulsion.dXdt:mass_fuel'])
        print('D', self.p['ode.aero.D'])
        print('thrust', self.p['ode.propulsion.thrust'])
        print('range rate', self.p['ode.range_rate_comp.dXdt:range'])

        from openmdao.api import view_model
        view_model(self.p.model)

        assert_rel_error(self,
                         self.p['ode.range_rate_comp.dXdt:range'],
                         self.p['TAS'] * np.cos(self.p['ode.gam_comp.gam']))

    # def test_partials(self):
    #     np.set_printoptions(linewidth=1024)
    #     cpd = self.p.check_partials(suppress_output=False)
    #     assert_check_partials(cpd, atol=1.0E-6, rtol=1.0)
