import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


def _run_brachistochrone(tol, refine_iteration_limit):
    """
    Solve the brachistochrone with grid refinement.

    Return the result of run_problem and the lines written to the grid refinement file.
    """
    p = om.Problem()
    p.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
    p.driver.options['disp'] = False
    p.driver.declare_coloring()

    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=dm.Radau(num_segments=3, order=3))
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10))
    phase.add_state('x', fix_initial=True)
    phase.add_state('y', fix_initial=True)
    phase.add_state('v', fix_initial=True)
    phase.add_control('theta', units='deg', lower=0.01, upper=179.9)
    phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)
    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    phase.add_objective('time', loc='final', scaler=10)

    phase.set_refine_options(refine=True, tol=tol)

    p.setup()

    phase.set_time_val(initial=0.0, duration=2.0)
    phase.set_state_val('x', [0, 10])
    phase.set_state_val('y', [10, 5])
    phase.set_state_val('v', [0, 9.9])
    phase.set_control_val('theta', [5, 100.5])

    result = dm.run_problem(p, refine_iteration_limit=refine_iteration_limit, simulate=False,
                            make_plots=False)

    with open('grid_refinement.out') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return result, lines


@use_tempdirs
class TestRefinementStatus(unittest.TestCase):

    # With this grid the largest error is about 1e-2 before the first refinement iteration,
    # 2e-6 after it and 2e-7 after the second one.

    def test_tolerance_met_before_iteration_limit(self):
        result, lines = _run_brachistochrone(tol=1.0E-4, refine_iteration_limit=3)

        # The error is checked twice: before and after the only refinement.
        self.assertEqual(sum('Grid Refinement - Iteration' in line for line in lines), 2)
        self.assertEqual(lines[-1], 'Successfully completed grid refinement.')
        self.assertIsNot(result, True)

    def test_tolerance_met_at_iteration_limit(self):
        result, lines = _run_brachistochrone(tol=1.0E-4, refine_iteration_limit=1)

        # The error on the grid of the last iteration is checked as well.
        self.assertEqual(sum('Grid Refinement - Iteration' in line for line in lines), 2)
        self.assertEqual(lines[-1], 'Successfully completed grid refinement.')
        self.assertIsNot(result, True)

    def test_tolerance_not_met_at_iteration_limit(self):
        result, lines = _run_brachistochrone(tol=1.0E-12, refine_iteration_limit=1)

        self.assertEqual(sum('Grid Refinement - Iteration' in line for line in lines), 2)
        self.assertEqual(lines[-1],
                         'Iteration limit exceeded. Unable to satisfy specified tolerance')
        self.assertIs(result, True)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
