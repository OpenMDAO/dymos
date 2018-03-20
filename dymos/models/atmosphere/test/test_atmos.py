from __future__ import print_function, division, absolute_import

import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from dymos.models.atmosphere import StandardAtmosphereGroup

assert_almost_equal = np.testing.assert_almost_equal

SHOW_PLOTS = False

if SHOW_PLOTS:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


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

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', subsys=IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output(name='alt_m', val=reference[:, 0], units='m')

        p.model.add_subsystem('atmos', subsys=StandardAtmosphereGroup(num_nodes=n))
        p.model.connect('alt_m', 'atmos.h')

        p.setup()
        p.run_model()

        var_map = ['alt_m', 'atmos.temp', 'atmos.pres', 'atmos.rho', 'atmos.sos']

        for i in range(1, 5):
            if SHOW_PLOTS:
                plt.plot(p['alt_m'], p[var_map[i]])
                plt.plot(reference[:, 0], reference[:, i], 'ro')
                plt.show()

        assert_rel_error(self, p['atmos.temp'], reference[:, 1], tolerance=1.0E-2)
        assert_rel_error(self, p['atmos.pres'], reference[:, 2], tolerance=1.0E-2)
        assert_rel_error(self, p['atmos.rho'], reference[:, 3], tolerance=1.0E-2)
        assert_rel_error(self, p['atmos.sos'], reference[:, 4], tolerance=1.0E-2)

if __name__ == "__main__":
    unittest.main()
