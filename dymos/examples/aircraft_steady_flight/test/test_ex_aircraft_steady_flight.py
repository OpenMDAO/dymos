from __future__ import print_function, absolute_import, division

import os
import unittest

from openmdao.utils.assert_utils import assert_rel_error

from dymos.examples.aircraft_steady_flight.ex_aircraft_steady_flight import \
    ex_aircraft_steady_flight


class TestExSteadyAircraftFlight(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['coloring.json', 'test_ex_aircraft_steady_flight_rec.db', 'SLSQP.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_ex_aircraft_steady_flight_opt(self):
        p = ex_aircraft_steady_flight(optimizer='SLSQP', transcription='gauss-lobatto',
                                      solve_segments=False)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:range', units='NM')[-1],
                         726.85, tolerance=1.0E-2)

    def test_ex_aircraft_steady_flight_solve(self):
        p = ex_aircraft_steady_flight(optimizer='SLSQP', transcription='gauss-lobatto',
                                      solve_segments=True, use_boundary_constraints=True)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:range', units='NM')[-1],
                         726.85, tolerance=1.0E-2)

if __name__ == '__main__':
    unittest.main()
