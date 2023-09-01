import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem


@require_pyoptsparse(optimizer='IPOPT')
@unittest.skipUnless(MPI, "MPI is required.")
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseMPI(unittest.TestCase):
    N_PROCS = 3

    def test_ex_two_burn_orbit_raise_mpi(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         show_output=False)

        sol_case = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case = om.CaseReader('dymos_simulation.db').get_case('final')

        assert_near_equal(sol_case.get_val('traj.burn2.timeseries.deltav')[-1], 0.3995, tolerance=2.0E-3)
        assert_near_equal(sim_case.get_val('traj.burn2.timeseries.deltav')[-1], 0.3995, tolerance=2.0E-3)

    def test_ex_two_burn_orbit_raise_connected_mpi(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=True, show_output=False)

        sol_case = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case = om.CaseReader('dymos_simulation.db').get_case('final')

        assert_near_equal(sol_case.get_val('traj.burn2.timeseries.deltav')[-1], 0.3995, tolerance=2.0E-3)
        assert_near_equal(sim_case.get_val('traj.burn2.timeseries.deltav')[-1], 0.3995, tolerance=2.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
