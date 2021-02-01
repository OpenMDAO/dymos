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

    def test_results(self):
        n = 10

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('mass_fuel', val=20000 * np.ones(n), units='kg',
                       desc='aircraft fuel mass')
        ivc.add_output('mass_payload', val=84.02869 * 400 * np.ones(n), units='kg',
                       desc='aircraft fuel mass')
        ivc.add_output('mass_empty', val=0.15E6 * np.ones(n), units='kg',
                       desc='aircraft empty mass')
        ivc.add_output('alt', val=9.144*np.ones(n), units='km', desc='altitude')
        ivc.add_output('climb_rate', val=np.zeros(n), units='m/s', desc='climb rate')
        ivc.add_output('mach', val=0.8*np.ones(n), units=None, desc='mach number')
        ivc.add_output('S', val=427.8 * np.ones(n), units='m**2', desc='reference area')

        p.model.add_subsystem('ode', AircraftODE(num_nodes=n))

        p.model.connect('mass_fuel', 'ode.mass_fuel')
        p.model.connect('mass_payload', 'ode.mass_comp.mass_payload')
        p.model.connect('mass_empty', 'ode.mass_comp.mass_empty')
        p.model.connect('alt', 'ode.alt')
        p.model.connect('mach', ['ode.tas_comp.mach', 'ode.aero.mach'])
        p.model.connect('climb_rate', ['ode.gam_comp.climb_rate'])
        p.model.connect('S', ('ode.aero.S', 'ode.flight_equilibrium.S', 'ode.propulsion.S'))

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p.run_model()

        assert_near_equal(p['ode.range_rate_comp.dXdt:range'],
                          p['ode.tas_comp.TAS'] * np.cos(p['ode.gam_comp.gam']))


if __name__ == "__main__":
    unittest.main()
