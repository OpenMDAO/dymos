import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from dymos.models.atmosphere.atmos_1976 import USatm1976Comp, USatm1976Data


class TestAtmosphere(unittest.TestCase):

    def test_atmos_comp_geopotential(self):
        n = USatm1976Data.alt.size

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='alt', val=USatm1976Data.alt, units='ft')

        p.model.add_subsystem('atmos', subsys=USatm1976Comp(num_nodes=n))
        p.model.connect('alt', 'atmos.h')

        p.setup(force_alloc_complex=True)
        p.run_model()

        T = p.get_val('atmos.temp', units='degR')
        P = p.get_val('atmos.pres', units='psi')
        rho = p.get_val('atmos.rho', units='slug/ft**3')
        sos = p.get_val('atmos.sos', units='ft/s')

        assert_near_equal(T, USatm1976Data.T, tolerance=1.0E-4)
        assert_near_equal(P, USatm1976Data.P, tolerance=1.0E-4)
        assert_near_equal(rho, USatm1976Data.rho, tolerance=1.0E-4)
        assert_near_equal(sos, USatm1976Data.a, tolerance=1.0E-4)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_atmos_comp_geodetic(self):
        n = USatm1976Data.alt.size

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='alt', val=USatm1976Data.alt, units='ft')

        p.model.add_subsystem('atmos', subsys=USatm1976Comp(num_nodes=n, h_def='geodetic'))
        p.model.connect('alt', 'atmos.h')

        p.setup(force_alloc_complex=True)

        h = USatm1976Data.alt * 0.3048  # altitude data in meters
        R0 = 6_356_766  # US 1976 std atm R0 in m
        p.set_val('alt', R0 / (R0 - h) * h, units='m')  # US 1976 std atm geopotential altitude to geodetic (m)

        p.run_model()

        T = p.get_val('atmos.temp', units='degR')
        P = p.get_val('atmos.pres', units='psi')
        rho = p.get_val('atmos.rho', units='slug/ft**3')
        sos = p.get_val('atmos.sos', units='ft/s')

        assert_near_equal(T, USatm1976Data.T, tolerance=1.0E-4)
        assert_near_equal(P, USatm1976Data.P, tolerance=1.0E-4)
        assert_near_equal(rho, USatm1976Data.rho, tolerance=1.0E-4)
        assert_near_equal(sos, USatm1976Data.a, tolerance=1.0E-4)

        with np.printoptions(linewidth=100000):
            cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
