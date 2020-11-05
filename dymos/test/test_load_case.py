import os
import unittest
from openmdao.utils.testing_utils import use_tempdirs
import openmdao
import openmdao.api as om
import dymos as dm

om_version = tuple([int(s) for s in openmdao.__version__.split('-')[0].split('.')])


def setup_problem(trans=dm.GaussLobatto(num_segments=10), polynomial_control=False):
    from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
    from dymos.transcriptions.runge_kutta.runge_kutta import RungeKutta

    p = om.Problem(model=om.Group())
    if isinstance(trans, RungeKutta):
        p.driver = om.pyOptSparseDriver()
    else:
        p.driver = om.ScipyOptimizeDriver()

    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=trans)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

    phase.add_state('x', fix_initial=True, fix_final=not isinstance(trans, RungeKutta),
                    rate_source=BrachistochroneODE.states['x']['rate_source'],
                    units=BrachistochroneODE.states['x']['units'])
    phase.add_state('y', fix_initial=True, fix_final=not isinstance(trans, RungeKutta),
                    rate_source=BrachistochroneODE.states['y']['rate_source'],
                    units=BrachistochroneODE.states['y']['units'])
    phase.add_state('v', fix_initial=True,
                    rate_source=BrachistochroneODE.states['v']['rate_source'],
                    units=BrachistochroneODE.states['v']['units'])

    if not polynomial_control:
        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)
    else:
        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

    if isinstance(trans, RungeKutta):
        phase.add_timeseries_output('check', units='m/s', shape=(1,))
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()

    p.setup()

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
    p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
    p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    if not polynomial_control:
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')
    else:
        p['phase0.polynomial_controls:theta'][:] = 5.0

    return p


@unittest.skipIf(om_version <= (2, 9, 0), 'load_case requires an OpenMDAO version later than 2.9.0')
@use_tempdirs
class TestLoadCase(unittest.TestCase):

    def tearDown(self):
        for filename in ['brachistochrone_solution.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_load_case_unchanged_grid(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.GaussLobatto(num_segments=10))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # Initialize the system with values from the case.
        # We unnecessarily call setup again just to make sure we obliterate the previous solution
        p.setup()

        # Load the values from the previous solution
        dm.load_case(p, case)

        # Run the model to ensure we find the same output values as those that we recorded
        p.run_driver()

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        assert_near_equal(p['phase0.controls:theta'],
                          outputs['phase0.control_group.indep_controls.controls:theta']['value'])

    def test_load_case_unchanged_grid_polynomial_control(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.GaussLobatto(num_segments=10), polynomial_control=True)

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # Initialize the system with values from the case.
        # We unnecessarily call setup again just to make sure we obliterate the previous solution
        p.setup()

        # Load the values from the previous solution
        dm.load_case(p, case)

        # Run the model to ensure we find the same output values as those that we recorded
        p.run_driver()

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        assert_near_equal(p['phase0.polynomial_controls:theta'],
                          outputs['phase0.polynomial_control_group.indep_polynomial_controls.polynomial_controls:theta']
                          ['value'])

    def test_load_case_lgl_to_radau(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.GaussLobatto(num_segments=10))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.Radau(num_segments=20))

        # Load the values from the previous solution
        dm.load_case(q, case)

        # Run the model to ensure we find the same output values as those that we recorded
        q.run_driver()

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        time_val = outputs['phase0.timeseries.time']['value']
        theta_val = outputs['phase0.timeseries.controls:theta']['value']

        assert_near_equal(q['phase0.timeseries.controls:theta'],
                          q.model.phase0.interpolate(xs=time_val, ys=theta_val, nodes='all'),
                          tolerance=1.0E-3)

    def test_load_case_radau_to_lgl(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.Radau(num_segments=20))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.GaussLobatto(num_segments=50))

        # Load the values from the previous solution
        dm.load_case(q, case)

        # Run the model to ensure we find the same output values as those that we recorded
        q.run_driver()

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        time_val = outputs['phase0.timeseries.time']['value']
        theta_val = outputs['phase0.timeseries.controls:theta']['value']

        assert_near_equal(q['phase0.timeseries.controls:theta'],
                          q.model.phase0.interpolate(xs=time_val, ys=theta_val, nodes='all'),
                          tolerance=1.0E-2)

    def test_load_case_rk4_to_lgl(self):
        import openmdao.api as om
        import dymos as dm
        from openmdao.utils.assert_utils import assert_near_equal

        p = setup_problem(dm.RungeKutta(num_segments=50))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.GaussLobatto(num_segments=10))

        # Load the values from the previous solution
        dm.load_case(q, case)

        # Run the model to ensure we find the same output values as those that we recorded
        q.run_driver()

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        time_val = outputs['phase0.timeseries.time']['value']
        theta_val = outputs['phase0.timeseries.controls:theta']['value']

        assert_near_equal(q['phase0.timeseries.controls:theta'],
                          q.model.phase0.interpolate(xs=time_val, ys=theta_val, nodes='all'),
                          tolerance=1.0E-1)

    def test_load_case_lgl_to_rk4(self):
        import openmdao.api as om
        import dymos as dm
        from openmdao.utils.assert_utils import assert_near_equal
        from scipy.interpolate import interp1d
        import numpy as np

        p = setup_problem(dm.GaussLobatto(num_segments=20))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.RungeKutta(num_segments=50))

        # Load the values from the previous solution
        dm.load_case(q, case)

        # Run the model to ensure we find the same output values as those that we recorded
        q.run_driver()

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        time_val = outputs['phase0.timeseries.time']['value'][:, 0]
        theta_val = outputs['phase0.timeseries.controls:theta']['value'][:, 0]
        nodup = np.insert(time_val[1:] != time_val[:-1], 0, True)  # remove duplicate times
        time_val = time_val[nodup]
        theta_val = theta_val[nodup]

        q_time = q['phase0.timeseries.time'][:, 0]
        q_theta = q['phase0.timeseries.controls:theta'][:, 0]
        nodup = np.insert(q_time[1:] != q_time[:-1], 0, True)  # remove duplicate times
        q_time = q_time[nodup]
        q_theta = q_theta[nodup]
        fq_theta = interp1d(q_time, q_theta, kind='cubic', bounds_error=False, fill_value='extrapolate')

        assert_near_equal(fq_theta(time_val),
                          theta_val,
                          tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
