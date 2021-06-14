import unittest
from numpy.testing import assert_almost_equal
import numpy as np

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI

import dymos as dm
from dymos.examples.vanderpol.vanderpol_dymos import vanderpol
from dymos.utils.testing_utils import require_pyoptsparse


@use_tempdirs
class TestVanderpolExample(unittest.TestCase):
    def test_vanderpol_simulate(self):
        # simulate only: with no control, the system oscillates
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)
        p.run_model()

    def test_vanderpol_optimal(self):
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)
        dm.run_problem(p)  # find optimal control solution to stop oscillation

        print('Objective function minimized to', p.get_val('traj.phase0.states:J')[-1, ...])
        # check that ODE states (excluding J) and control are driven to near zero
        assert_almost_equal(p.get_val('traj.phase0.states:x0')[-1, ...], np.zeros(1))
        assert_almost_equal(p.get_val('traj.phase0.states:x1')[-1, ...], np.zeros(1))
        assert_almost_equal(p.get_val('traj.phase0.controls:u')[-1, ...], np.zeros(1), decimal=3)

    def test_vanderpol_optimal_grid_refinement(self):
        # enabling grid refinement gives a faster and better solution with fewer segments
        p = vanderpol(transcription='gauss-lobatto', num_segments=15)

        p.model.traj.phases.phase0.set_refine_options(refine=True)
        dm.run_problem(p, refine_iteration_limit=10)  # enable grid refinement and find optimal solution

        print('Objective function minimized to', p.get_val('traj.phase0.timeseries.states:J')[-1, ...])
        # check that ODE states (excluding J) and control are driven to near zero
        assert_almost_equal(p.get_val('traj.phase0.timeseries.states:x0')[-1, ...], np.zeros(1))
        assert_almost_equal(p.get_val('traj.phase0.timeseries.states:x1')[-1, ...], np.zeros(1))
        assert_almost_equal(p.get_val('traj.phase0.timeseries.controls:u')[-1, ...], np.zeros(1), decimal=4)


@use_tempdirs
class TestVanderpolExampleMPI(unittest.TestCase):

    N_PROCS = 4

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipUnless(MPI, 'this test requires MPI')
    def test_vanderpol_optimal_mpi(self):
        """to test with MPI:
           OPENMDAO_REQUIRE_MPI=1 mpirun -n 4 python
               dymos/examples/vanderpol/test/test_vanderpol.py TestVanderpolExampleMPI.test_vanderpol_optimal_mpi
           (using varying values for n should give the same answer)
        """
        p = vanderpol(transcription='gauss-lobatto', num_segments=75, delay=0.005,
                      use_pyoptsparse=True, distrib=True, optimizer='IPOPT')
        p.run_driver()  # find optimal control solution to stop oscillation

        print('Objective function minimized to', p.get_val('traj.phase0.states:J')[-1, ...])
        # check that ODE states (excluding J) and control are driven to near zero
        assert_almost_equal(p.get_val('traj.phase0.states:x0')[-1, ...], np.zeros(1))
        assert_almost_equal(p.get_val('traj.phase0.states:x1')[-1, ...], np.zeros(1))
        assert_almost_equal(p.get_val('traj.phase0.controls:u')[-1, ...], np.zeros(1), decimal=3)

if __name__ == '__main__':  # pragma: no cover
    unittest.main()
