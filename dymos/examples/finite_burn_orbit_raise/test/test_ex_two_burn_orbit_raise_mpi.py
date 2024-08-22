import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem
from dymos.utils.misc import om_version


@require_pyoptsparse(optimizer='IPOPT')
@unittest.skipUnless(MPI, "MPI is required.")
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseMPI(unittest.TestCase):
    N_PROCS = 3

    def test_ex_two_burn_orbit_raise_mpi(self):
        optimizer = 'IPOPT'

        CONNECTED = False

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=False)

        sol_db = 'dymos_solution.db'
        sim_db = 'dymos_simulation.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db
            sim_db = p.model.traj.sim_prob.get_outputs_dir() / sim_db

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')

        # The last phase in this case is run in reverse time if CONNECTED=True,
        # so grab the correct index to test the resulting delta-V.
        end_idx = 0 if CONNECTED else -1

        assert_near_equal(sol_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)
        assert_near_equal(sim_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)

    def test_ex_two_burn_orbit_raise_connected_mpi(self):
        optimizer = 'IPOPT'

        CONNECTED = True

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=False)

        sol_db = 'dymos_solution.db'
        sim_db = 'dymos_simulation.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db
            sim_db = p.model.traj.sim_prob.get_outputs_dir() / sim_db

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')

        # The last phase in this case is run in reverse time if CONNECTED=True,
        # so grab the correct index to test the resulting delta-V.
        end_idx = 0 if CONNECTED else -1

        assert_near_equal(sol_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)
        assert_near_equal(sim_case.get_val('traj.burn2.timeseries.deltav')[end_idx], 0.3995, tolerance=2.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
