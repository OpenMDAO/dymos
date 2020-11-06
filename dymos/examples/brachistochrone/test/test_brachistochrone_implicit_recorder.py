import os
import unittest
from openmdao.utils.testing_utils import use_tempdirs


class TestBrachistochroneRecordingExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['brachistochrone_solution.db']:
            if os.path.exists(filename):
                os.remove(filename)

    @use_tempdirs
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

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['x']['rate_source'])
        phase.add_state('y', fix_initial=True, fix_final=True,
                        rate_source=BrachistochroneODE.states['y']['rate_source'])
        phase.add_state('v', fix_initial=True,
                        rate_source=BrachistochroneODE.states['v']['rate_source'])

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        case = om.CaseReader('dymos_solution.db').get_case('final')

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        assert_near_equal(p['phase0.controls:theta'],
                          outputs['phase0.control_group.indep_controls.controls:theta']['value'])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
