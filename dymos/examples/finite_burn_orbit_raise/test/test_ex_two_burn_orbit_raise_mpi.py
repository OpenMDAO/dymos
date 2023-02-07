import unittest

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
                                         compressed=False, optimizer=optimizer, simulate=False,
                                         show_output=False)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)

    def test_ex_two_burn_orbit_raise_connected_mpi(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer, simulate=False,
                                         connected=True, show_output=False)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                              tolerance=2.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
