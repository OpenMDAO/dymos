import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

assert_almost_equal = np.testing.assert_almost_equal

reference = np.array([[0.00000, 288.150, 101325., 1.22500,  340.294, 0.0000181206],
                      [1000.00, 281.650, 89874.6, 1.111640, 336.434, 0.0000177943],
                      [2000.00, 275.150, 79495.2, 1.006490, 332.529, 0.0000174645],
                      [3000.00, 268.650, 70108.5, 0.909122, 328.578, 0.0000171311],
                      [4000.00, 262.150, 61640.2, 0.819129, 324.579, 0.0000167940],
                      [5000.00, 255.650, 54019.9, 0.736116, 320.529, 0.0000164531],
                      [6000.00, 249.150, 47181.0, 0.659697, 316.428, 0.0000161084],
                      [7000.00, 242.650, 41060.7, 0.589501, 312.274, 0.0000157596],
                      [8000.00, 236.150, 35599.8, 0.525168, 308.063, 0.0000154068],
                      [9000.00, 229.650, 30742.5, 0.466348, 303.793, 0.0000150498],
                      [10000.0, 223.150, 26436.3, 0.412707, 299.463, 0.0000146884],
                      [11000.0, 216.650, 22632.1, 0.363918, 295.070, 0.0000143226],
                      [12000.0, 216.650, 19330.4, 0.310828, 295.070, 0.0000143226],
                      [13000.0, 216.650, 16510.4, 0.265483, 295.070, 0.0000143226],
                      [14000.0, 216.650, 14101.8, 0.226753, 295.070, 0.0000143226],
                      [15000.0, 216.650, 12044.6, 0.193674, 295.070, 0.0000143226],
                      [16000.0, 216.650, 10287.5, 0.165420, 295.070, 0.0000143226],
                      [17000.0, 216.650, 8786.68, 0.141288, 295.070, 0.0000143226],
                      [18000.0, 216.650, 7504.84, 0.120676, 295.070, 0.0000143226],
                      [19000.0, 216.650, 6410.01, 0.103071, 295.070, 0.0000143226],
                      [20000.0, 216.650, 5474.89, 0.0880349, 295.070, 0.0000143226]])


class TestAtmosphere(unittest.TestCase):

    def test_temperature_comp(self):
        n = reference.shape[0]

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='alt_m', val=reference[:, 0], units='m')

        p.model.add_subsystem('atmos', subsys=USatm1976Comp(num_nodes=n))
        p.model.connect('alt_m', 'atmos.h')

        p.setup()
        p.run_model()

        assert_near_equal(p.get_val('atmos.temp', units='K'),
                          reference[:, 1], tolerance=1.0E-2)
        assert_near_equal(p.get_val('atmos.pres', units='Pa'),
                          reference[:, 2], tolerance=1.0E-2)
        assert_near_equal(p.get_val('atmos.rho', units='kg/m**3'),
                          reference[:, 3], tolerance=1.0E-2)
        assert_near_equal(p.get_val('atmos.sos', units='m/s'),
                          reference[:, 4], tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
