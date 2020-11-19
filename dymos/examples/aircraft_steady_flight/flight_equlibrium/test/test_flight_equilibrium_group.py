import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.testing_utils import assert_check_partials

from dymos.examples.aircraft_steady_flight.flight_equlibrium.steady_flight_equilibrium_group \
    import SteadyFlightEquilibriumGroup


try:
    import MBI
except:
    MBI = None


@unittest.skipIf(MBI is None, 'MBI not available')
class TestFlightEquilibriumGroup(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 10

        cls.p = om.Problem(model=om.Group())

        ivc = cls.p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('alt', val=3000 * np.ones(cls.n), units='m', desc='altitude above MSL')
        ivc.add_output('TAS', val=250.0 * np.ones(cls.n), units='m/s', desc='true airspeed')
        # ivc.add_output('TAS_rate', val=np.zeros(cls.n), units='m/s**2', desc='acceleration')
        ivc.add_output('gam', val=np.zeros(cls.n), units='rad', desc='flight path angle')
        # ivc.add_output('gam_rate', val=np.zeros(cls.n), units='rad/s',
        #                desc='flight path angle rate')
        ivc.add_output('S', val=427.8 * np.ones(cls.n), units='m**2', desc='reference area')
        ivc.add_output('mach', val=0.8 * np.ones(cls.n), units=None, desc='mach number')
        ivc.add_output('W_total', val=200000 * 9.80665 * np.ones(cls.n), units='N',
                       desc='aircraft total weight')
        ivc.add_output('q', val=0.5 * 250.0**2 * np.ones(cls.n), units='Pa',
                       desc='dynamic pressure')

        cls.p.model.add_subsystem('flight_equilibrium',
                                  subsys=SteadyFlightEquilibriumGroup(num_nodes=cls.n),
                                  promotes_inputs=['aero.*'],
                                  promotes_outputs=['aero.*'])

        cls.p.model.connect('gam', 'flight_equilibrium.gam')
        cls.p.model.connect('S', ('aero.S', 'flight_equilibrium.S'))
        cls.p.model.connect('mach', 'aero.mach')
        cls.p.model.connect('W_total', 'flight_equilibrium.W_total')
        cls.p.model.connect('q', ('aero.q', 'flight_equilibrium.q'))

        cls.p.setup(check=True, force_alloc_complex=True)

        cls.p.run_model()

    def test_results(self):
        CL_eq = self.p['flight_equilibrium.CL_eq']
        CL = self.p['aero.CL']
        CM = self.p['aero.CM']

        assert_near_equal(CL_eq, CL, tolerance=1.0E-12)
        assert_near_equal(CM, np.zeros_like(CM), tolerance=1.0E-12)

    def test_partials(self):
        cpd = self.p.check_partials(out_stream=None, method='fd', step=1.0E-6)
        assert_check_partials(cpd, atol=5.0E-3, rtol=2.0)
