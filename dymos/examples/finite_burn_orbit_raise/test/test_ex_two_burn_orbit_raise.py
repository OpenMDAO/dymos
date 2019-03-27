from __future__ import print_function, division, absolute_import

import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp
import errno

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import set_pyoptsparse_opt

from dymos.examples.finite_burn_orbit_raise.ex_finite_burn_orbit_raise import \
    two_burn_orbit_raise_problem


class TestExampleTwoBurnOrbitRaise(unittest.TestCase):

    def setUp(self):
        self.orig_dir = os.getcwd()
        self.temp_dir = mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.orig_dir)
        try:
            rmtree(self.temp_dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_ex_two_burn_orbit_raise(self):
        _, optimizer = set_pyoptsparse_opt('SNOPT', fallback=False)

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=True, show_plots=False, optimizer=optimizer,
                                         show_output=False)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_rel_error(self, p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                             tolerance=2.0E-3)


# This test is separate because connected phases aren't directly parallelizable.
class TestExampleTwoBurnOrbitRaiseConnected(unittest.TestCase):

    def setUp(self):
        self.orig_dir = os.getcwd()
        self.temp_dir = mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.orig_dir)
        try:
            rmtree(self.temp_dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_ex_two_burn_orbit_raise_connected(self):
        _, optimizer = set_pyoptsparse_opt('SNOPT', fallback=False)

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=True, show_plots=False, optimizer=optimizer,
                                         show_output=False, connected=True)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_rel_error(self, p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                             tolerance=2.0E-3)


class TestExampleTwoBurnOrbitRaiseMPI(TestExampleTwoBurnOrbitRaise):
    N_PROCS = 3


if __name__ == '__main__':
    unittest.main()
