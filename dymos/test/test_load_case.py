import os
import unittest
from openmdao.utils.testing_utils import use_tempdirs
import openmdao
import openmdao.api as om
import dymos as dm

om_version = tuple([int(s) for s in openmdao.__version__.split('-')[0].split('.')])


def setup_problem(trans=dm.GaussLobatto(num_segments=10), polynomial_control=False,
                  fix_final_state=True, fix_final_control=False):
    from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

    p = om.Problem(model=om.Group())
    p.driver = om.ScipyOptimizeDriver()

    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=trans)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.add_state('x', fix_initial=True, fix_final=fix_final_state)
    phase.add_state('y', fix_initial=True, fix_final=fix_final_state)
    phase.add_state('v', fix_initial=True)

    if not polynomial_control:
        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9, fix_final=fix_final_control)
    else:
        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9,
                                     fix_final=fix_final_control)

    phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()

    p.setup()

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    p['phase0.states:x'] = phase.interp('x', [0, 10])
    p['phase0.states:y'] = phase.interp('y', [10, 5])
    p['phase0.states:v'] = phase.interp('v', [0, 9.9])

    if polynomial_control:
        p['phase0.polynomial_controls:theta'] = phase.interp('theta', [5, 100.5])
    else:
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100.5])

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

        inputs = dict([(o[0], o[1]) for o in case.list_inputs(units=True, shape=True, out_stream=None)])

        assert_near_equal(p['phase0.controls:theta'],
                          inputs['phase0.control_group.control_interp_comp.controls:theta']
                          ['value'])

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

        inputs = dict([(o[0], o[1]) for o in case.list_inputs(units=True, shape=True, out_stream=None)])

        assert_near_equal(p['phase0.polynomial_controls:theta'],
                          inputs['phase0.polynomial_control_group.interp_comp.polynomial_controls:theta']
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

        print(outputs)
        time_val = outputs['phase0.timeseries.timeseries_comp.time']['val']
        theta_val = outputs['phase0.timeseries.timeseries_comp.controls:theta']['val']

        assert_near_equal(q['phase0.timeseries.timeseries_comp.controls:theta'],
                          q.model.phase0.interp(xs=time_val, ys=theta_val, nodes='all'),
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

        time_val = outputs['phase0.timeseries.timeseries_comp.time']['val']
        theta_val = outputs['phase0.timeseries.timeseries_comp.controls:theta']['val']

        assert_near_equal(q['phase0.timeseries.timeseries_comp.controls:theta'],
                          q.model.phase0.interp(xs=time_val, ys=theta_val, nodes='all'),
                          tolerance=1.0E-2)

    def test_load_case_warn_fix_final_states(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_warnings
        import dymos as dm

        p = setup_problem(dm.Radau(num_segments=20))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.GaussLobatto(num_segments=50))

        msgs = []

        # Load the values from the previous solution
        for state_name in ['x', 'y']:
            msgs.append((UserWarning, f"phase0.states:{state_name} specifies 'fix_final=True'."
                                      f" If the given restart file has a different final value"
                                      f" this will overwrite the user-specified value"))

        with assert_warnings(msgs):
            dm.load_case(q, case)

    def test_load_case_warn_fix_final_control(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_warning
        import dymos as dm
        p = setup_problem(dm.Radau(num_segments=10))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.Radau(num_segments=10), fix_final_state=False, fix_final_control=True)

        msg = f"phase0.controls:theta specifies 'fix_final=True'. If the given restart file has a" \
              f" different final value this will overwrite the user-specified value"

        with assert_warning(UserWarning, msg):
            dm.load_case(q, case)

    def test_load_case_warn_fix_final_polynomial_control(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_warning
        import dymos as dm
        p = setup_problem(dm.Radau(num_segments=10), polynomial_control=True,)

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        case = om.CaseReader('dymos_solution.db').get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.Radau(num_segments=10), polynomial_control=True,
                          fix_final_state=False, fix_final_control=True)

        # Load the values from the previous solution
        msg = f"phase0.polynomial_controls:theta specifies 'fix_final=True'. If the given restart file has a" \
              f" different final value this will overwrite the user-specified value"

        with assert_warning(UserWarning, msg):
            dm.load_case(q, case)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
