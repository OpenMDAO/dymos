from __future__ import print_function, division, absolute_import

import os
import unittest

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import set_pyoptsparse_opt

from dymos.examples.finite_burn_orbit_raise.ex_two_burn_orbit_raise import \
    two_burn_orbit_raise_problem


class TestExampleTwoBurnOrbitRaise(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['coloring.json', 'two_burn_orbit_raise_example.db', 'SLSQP.out',
                         'traj_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_ex_two_burn_orbit_raise(self):
        _, optimizer = set_pyoptsparse_opt('SNOPT', fallback=False)

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=True, show_plots=False, optimizer=optimizer)

        assert_rel_error(self, p.get_val('burn2.states:deltav')[-1], 0.3995, tolerance=2.0E-3)


if __name__ == '__main__':
    unittest.main()
