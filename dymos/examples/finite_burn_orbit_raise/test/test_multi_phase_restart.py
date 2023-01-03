import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
import scipy

from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem
from dymos.utils.testing_utils import assert_cases_equal


# This test is separate because connected phases aren't directly parallelizable.
@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseConnectedRestart(unittest.TestCase):

    @unittest.skip('Skipped due to a change in interpolation in scipy. Need to come up with better case loading.')
    def test_ex_two_burn_orbit_raise_connected(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                              tolerance=4.0E-3)

        case1 = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case1 = om.CaseReader('dymos_simulation.db').get_case('final')

        # Run again without an actual optimizer
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True, run_driver=False,
                                         restart='dymos_solution.db')

        p.run_model()

        sim_case2 = om.CaseReader('dymos_simulation.db').get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(case1, p, tol=1.0E-8)
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-8)

    def test_restart_from_solution_radau(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer, show_output=False)

        case1 = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case1 = om.CaseReader('dymos_simulation.db').get_case('final')

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)

        # Run again without an actual optimzier
        two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                     compressed=False, optimizer=optimizer, run_driver=False,
                                     show_output=False, restart='dymos_solution.db')

        sim_case2 = om.CaseReader('dymos_simulation.db').get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(case1, p, tol=1.0E-9)
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-8)


# This test is separate because connected phases aren't directly parallelizable.
@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseConnected(unittest.TestCase):

    @unittest.skip('Skipped due to a change in interpolation in scipy. Need to come up with better case loading.')
    def test_ex_two_burn_orbit_raise_connected(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                              tolerance=4.0E-3)

        case1 = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case1 = om.CaseReader('dymos_simulation.db').get_case('final')

        # Run again without an actual optimizer
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, run_driver=False,
                                         show_output=False, restart='dymos_solution.db',
                                         connected=True)

        sim_case2 = om.CaseReader('dymos_simulation.db').get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(case1, p, tol=1.0E-8)
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-8)

    @unittest.skip('Skipped due to a change in interpolation in scipy. Need to come up with better case loading.')
    def test_restart_from_solution_radau_to_connected(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                         compressed=False, optimizer=optimizer, show_output=False)

        case1 = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case1 = om.CaseReader('dymos_simulation.db').get_case('final')

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)

        # Run again without an actual optimzier
        two_burn_orbit_raise_problem(transcription='radau', transcription_order=3,
                                     compressed=False, optimizer=optimizer, run_driver=False,
                                     show_output=False, restart='dymos_solution.db', connected=True)

        sim_case2 = om.CaseReader('dymos_simulation.db').get_case('final')

        # Verify that the second case has the same inputs and outputs
        assert_cases_equal(case1, p, tol=1.0E-9, require_same_vars=False)
        assert_cases_equal(sim_case1, sim_case2, tol=1.0E-8)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
