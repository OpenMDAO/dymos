import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE


try:
    import MBI
except:
    MBI = None


@unittest.skipIf(MBI is None, 'MBI not available')
@use_tempdirs
class TestAircraftODEGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 10

        cls.p = om.Problem(model=om.Group())

        ivc = cls.p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('mass_fuel', val=20000 * np.ones(cls.n), units='kg',
                       desc='aircraft fuel mass')
        ivc.add_output('mass_payload', val=84.02869 * 400 * np.ones(cls.n), units='kg',
                       desc='aircraft fuel mass')
        ivc.add_output('mass_empty', val=0.15E6 * np.ones(cls.n), units='kg',
                       desc='aircraft empty mass')
        ivc.add_output('alt', val=9.144*np.ones(cls.n), units='km', desc='altitude')
        ivc.add_output('climb_rate', val=np.zeros(cls.n), units='m/s', desc='climb rate')
        ivc.add_output('mach', val=0.8*np.ones(cls.n), units=None, desc='mach number')
        ivc.add_output('S', val=427.8 * np.ones(cls.n), units='m**2', desc='reference area')

        cls.p.model.add_subsystem('ode', AircraftODE(num_nodes=cls.n))

        cls.p.model.connect('mass_fuel', 'ode.mass_fuel')
        cls.p.model.connect('mass_payload', 'ode.mass_comp.mass_payload')
        cls.p.model.connect('mass_empty', 'ode.mass_comp.mass_empty')
        cls.p.model.connect('alt', 'ode.alt')
        cls.p.model.connect('mach', ['ode.tas_comp.mach', 'ode.aero.mach'])
        cls.p.model.connect('climb_rate', ['ode.gam_comp.climb_rate'])
        cls.p.model.connect('S', ('ode.aero.S', 'ode.flight_equilibrium.S', 'ode.propulsion.S'))

        cls.p.model.linear_solver = om.DirectSolver()

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p.run_model()

    def test_results(self):
        assert_near_equal(self.p['ode.range_rate_comp.dXdt:range'],
                          self.p['ode.tas_comp.TAS'] * np.cos(self.p['ode.gam_comp.gam']))


if __name__ == "__main__":
    unittest.main()
