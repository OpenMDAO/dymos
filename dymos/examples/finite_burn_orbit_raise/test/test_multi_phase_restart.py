import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.mpi import MPI, multi_proc_exception_check

from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem
from dymos.utils.testing_utils import assert_cases_equal


@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseConnectedRestart(unittest.TestCase):

    N_PROCS = 3

    @unittest.skipUnless(MPI, "MPI is required.")
    def test_ex_two_burn_orbit_raise_connected(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True)

        with multi_proc_exception_check(p.comm):
            if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
                assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                                  tolerance=4.0E-3)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case1 = om.CaseReader(sim_db).get_case('final')

        # Run again without an actual optimizer
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True, run_driver=False,
                                         restart=sol_db)

        p.run_model()

        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case2 = om.CaseReader(sim_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)

    def test_restart_from_solution_radau(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer, show_output=False)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case1 = om.CaseReader(sim_db).get_case('final')

        with multi_proc_exception_check(p.comm):
            if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
                assert_near_equal(p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                                  tolerance=2.0E-3)

        # Run again without an optimzier
        two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                     compressed=False, optimizer=optimizer, run_driver=False,
                                     show_output=False, restart=sol_db)

        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sim_case2 = om.CaseReader(sim_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)


@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseConnected(unittest.TestCase):

    N_PROCS = 3

    @unittest.skipUnless(MPI, "MPI is required.")
    def test_ex_two_burn_orbit_raise_connected(self):
        optimizer = 'IPOPT'

        unexpected_warnings = \
            [(om.OpenMDAOWarning,
              "'traj' <class Trajectory>: Setting phases.nonlinear_solver to `om.NonlinearBlockJac(iprint=0)`.\n"
              "Connected phases in parallel require a non-default nonlinear solver.\n"
              "Use traj.options[\'default_nonlinear_solver\'] to explicitly set the solver."),
             (om.OpenMDAOWarning,
              "'traj' <class Trajectory>: Setting phases.linear_solver to `om.PETScKrylov()`.\n"
              "Connected phases in parallel require a non-default linear solver.\n"
              "Use traj.options[\'default_linear_solver\'] to explicitly set the solver.")]

        with warnings.catch_warnings(record=True) as w:
            p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                             compressed=False, optimizer=optimizer,
                                             show_output=False, connected=True,
                                             default_nonlinear_solver=om.NonlinearBlockJac(iprint=0),
                                             default_linear_solver=om.PETScKrylov())

        for category, msg in unexpected_warnings:
            for warn in w:
                if (issubclass(warn.category, category) and str(warn.message) == msg):
                    raise AssertionError(f"Saw unexpected warning {category.__name__}: {msg}")

        with multi_proc_exception_check(p.comm):
            if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
                assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                                  tolerance=4.0E-3)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        case1 = om.CaseReader(sol_db).get_case('final')
        sim_case1 = om.CaseReader(sim_db).get_case('final')

        # Run again without an actual optimizer
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, run_driver=False,
                                         show_output=False, restart=sol_db,
                                         connected=True, solution_record_file='dymos_solution2.db',
                                         simulation_record_file='dymos_simulation2.db')

        sol_db = p.get_outputs_dir() / 'dymos_solution2.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation2.db'

        case2 = om.CaseReader(sol_db).get_case('final')
        sim_case2 = om.CaseReader(sim_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(case1, case2, tol=1.0E-7)
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)

    @unittest.skipUnless(MPI, "MPI is required.")
    def test_restart_from_solution_radau_to_connected(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True)

        with multi_proc_exception_check(p.comm):
            if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
                assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                                  tolerance=4.0E-3)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        case1 = om.CaseReader(sol_db).get_case('final')
        sim_case1 = om.CaseReader(sim_db).get_case('final')

        # Run again without an actual optimizer
        p2 = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                          compressed=False, optimizer=optimizer, run_driver=False,
                                          show_output=False, restart=sol_db,
                                          connected=True, solution_record_file='dymos_solution2.db',
                                          simulation_record_file='dymos_simulation2.db')

        sol_db = p2.get_outputs_dir() / 'dymos_solution2.db'
        sim_db = p2.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation2.db'

        case2 = om.CaseReader(sol_db).get_case('final')
        sim_case2 = om.CaseReader(sim_db).get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(case1, case2, tol=1.0E-7)
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-7)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
