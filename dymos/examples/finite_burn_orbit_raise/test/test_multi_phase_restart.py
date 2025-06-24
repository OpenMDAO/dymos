import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.mpi import MPI, multi_proc_exception_check

from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem
from dymos.utils.misc import om_version
from dymos.utils.testing_utils import assert_cases_equal


@require_pyoptsparse(optimizer='IPOPT')
@unittest.skipUnless(MPI, "MPI is required.")
@unittest.skipIf(om_version()[0] < (3, 40, 0), "Test requires OpenMDAO 3.40.0 or later.")
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseMPI(unittest.TestCase):
    N_PROCS = 3

    def test_ex_two_burn_orbit_raise_mpi_restart_radau(self):
        optimizer = 'IPOPT'

        CONNECTED = False

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case1 = sim_case1 = om.CaseReader(sim_db).get_case('final')

        p2 = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                          compressed=False, simulate=True,
                                          connected=CONNECTED, run_driver=False, show_output=True,
                                          restart=sol_db)

        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'
        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case2 = om.CaseReader(sim2_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)

    def test_ex_two_burn_orbit_raise_mpi_restart_gl(self):
        optimizer = 'IPOPT'

        CONNECTED = False

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case1 = sim_case1 = om.CaseReader(sim_db).get_case('final')

        p2 = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                          compressed=False, simulate=True,
                                          connected=CONNECTED, run_driver=False, show_output=True,
                                          restart=sol_db)

        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'
        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case2 = om.CaseReader(sim2_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)

    def test_ex_two_burn_orbit_raise_mpi_restart_radau_connected(self):
        optimizer = 'IPOPT'

        CONNECTED = True

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case1 = sim_case1 = om.CaseReader(sim_db).get_case('final')

        p2 = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                          compressed=False, simulate=True,
                                          connected=CONNECTED, run_driver=False, show_output=True,
                                          restart=sol_db)

        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'
        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case2 = om.CaseReader(sim2_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)

    def test_ex_two_burn_orbit_raise_mpi_restart_gl_connected(self):
        optimizer = 'IPOPT'

        CONNECTED = True

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=True,
                                         connected=CONNECTED, show_output=True, run_driver=False,
                                         default_nonlinear_solver=om.NonlinearBlockJac(),
                                         default_linear_solver=om.ScipyKrylov(rhs_checking=True))

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case1 = sim_case1 = om.CaseReader(sim_db).get_case('final')

        p2 = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                          compressed=False, simulate=True,
                                          connected=CONNECTED, run_driver=False, show_output=True,
                                          restart=sol_db)

        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'
        sim2_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case2 = om.CaseReader(sim2_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
