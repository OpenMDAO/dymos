import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.finite_burn_orbit_raise.finite_burn_orbit_raise_problem import two_burn_orbit_raise_problem


@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestExampleTwoBurnOrbitRaise(unittest.TestCase):

    def test_ex_two_burn_orbit_raise(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)


# This test is separate because connected phases aren't directly parallelizable.
@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseConnected(unittest.TestCase):

    def test_ex_two_burn_orbit_raise_connected(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True, run_driver=True)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                              tolerance=4.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
