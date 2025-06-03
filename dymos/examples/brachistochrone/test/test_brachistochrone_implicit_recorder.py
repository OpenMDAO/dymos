import os
import unittest

try:
    import matplotlib
except ImportError:
    matplotlib = None

from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestBrachistochroneRecordingExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['brachistochrone_solution.db']:
            if os.path.exists(filename):
                os.remove(filename)

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_recording(self):
        import matplotlib
        matplotlib.use('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 9.9], units='m/s')

        phase.set_control_val('theta', [5, 100.5], units='deg')

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        case = om.CaseReader(sol_db).get_case('final')

        assert_near_equal(p['phase0.control_values:theta'],
                          case.get_val('phase0.timeseries.theta'))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
