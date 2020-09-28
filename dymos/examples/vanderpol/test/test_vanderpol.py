import unittest
from numpy.testing import assert_almost_equal
import numpy as np
import dymos as dm
from dymos.examples.vanderpol.vanderpol_dymos import vanderpol
from dymos.examples.vanderpol.vanderpol_dymos_plots import vanderpol_dymos_plots
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI

SHOW_PLOTS = False


@use_tempdirs
class TestVanderpolExample(unittest.TestCase):
    def test_vanderpol_simulate(self):
        # simulate only: with no control, the system oscillates
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)
        p.run_model()

        if SHOW_PLOTS:
            vanderpol_dymos_plots(p)

    def test_vanderpol_optimal(self):
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)
        dm.run_problem(p)  # find optimal control solution to stop oscillation

        if SHOW_PLOTS:
            vanderpol_dymos_plots(p)

        print('Objective function minimized to', p['traj.phases.phase0.final_conditions.states:J++'])
        # check that ODE states (excluding J) and control are driven to near zero
        assert_almost_equal(p['traj.phases.phase0.final_conditions.states:x0++'], np.zeros(1))
        assert_almost_equal(p['traj.phases.phase0.final_conditions.states:x1++'], np.zeros(1))
        assert_almost_equal(p['traj.phases.phase0.final_conditions.controls:u++'], np.zeros(1), decimal=3)

    @unittest.skipUnless(MPI, 'the test runs without MPI, but is very slow')
    def test_vanderpol_optimal_slow(self):
        """to test with MPI:
           OPENMDAO_REQUIRE_MPI=1 mpirun -n 4 python
               dymos/examples/vanderpol/test/test_vanderpol.py TestVanderpolExample.test_vanderpol_optimal_slow
           (using varying values for n should give the same answer)
        """
        p = vanderpol(transcription='gauss-lobatto', num_segments=75, delay=True)
        dm.run_problem(p)  # find optimal control solution to stop oscillation

        if SHOW_PLOTS:
            vanderpol_dymos_plots(p)

        print('Objective function minimized to', p['traj.phases.phase0.final_conditions.states:J++'])
        # check that ODE states (excluding J) and control are driven to near zero
        assert_almost_equal(p['traj.phases.phase0.final_conditions.states:x0++'], np.zeros(1))
        assert_almost_equal(p['traj.phases.phase0.final_conditions.states:x1++'], np.zeros(1))
        assert_almost_equal(p['traj.phases.phase0.final_conditions.controls:u++'], np.zeros(1), decimal=3)

    def test_vanderpol_optimal_grid_refinement(self):
        # enabling grid refinement gives a faster and better solution with fewer segments
        p = vanderpol(transcription='gauss-lobatto', num_segments=15)

        p.model.traj.phases.phase0.set_refine_options(refine=True)
        dm.run_problem(p, refine_iteration_limit=10)  # enable grid refinement and find optimal solution

        if SHOW_PLOTS:
            vanderpol_dymos_plots(p)

        print('Objective function minimized to', p['traj.phases.phase0.final_conditions.states:J++'])
        # check that ODE states (excluding J) and control are driven to near zero
        assert_almost_equal(p['traj.phases.phase0.final_conditions.states:x0++'], np.zeros(1))
        assert_almost_equal(p['traj.phases.phase0.final_conditions.states:x1++'], np.zeros(1))
        assert_almost_equal(p['traj.phases.phase0.final_conditions.controls:u++'], np.zeros(1), decimal=4)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
