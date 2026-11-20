import os
import unittest

from openmdao.utils.testing_utils import use_tempdirs
import openmdao.api as om
import dymos as dm


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
        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9,
                          fix_final=fix_final_control, control_type='polynomial')

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
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100.5])
    else:
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100.5])

    return p


class _ConstantRateODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('units', types=str)

    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']
        self.add_input('x', shape=(nn,), units=units)
        self.add_input('u', shape=(nn,), units=units)
        self.add_output('x_dot', shape=(nn,), units=f'{units}/s')

    def compute(self, inputs, outputs):
        outputs['x_dot'] = 1.0E-3


def setup_bounded_problem(units, timeseries_units, lower, upper, x0, xf, num_segments=4):
    """
    Build a phase with a bounded state and a bounded control that vary linearly from x0 to xf.

    The state, the control and their bounds are in `units`. The timeseries records the state
    and the control in `timeseries_units`.
    """
    p = om.Problem()

    phase = dm.Phase(ode_class=_ConstantRateODE, ode_init_kwargs={'units': units},
                     transcription=dm.Radau(num_segments=num_segments, order=3))

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, fix_duration=True, units='s')
    phase.add_state('x', rate_source='x_dot', units=units, fix_initial=True,
                    lower=lower, upper=upper)
    phase.add_control('u', units=units, lower=lower, upper=upper, opt=False)

    phase.add_timeseries_output('x', units=timeseries_units)
    phase.add_timeseries_output('u', units=timeseries_units)

    p.setup()

    phase.set_time_val(initial=0.0, duration=50.0)
    phase.set_state_val('x', [x0, xf])
    phase.set_control_val('u', [x0, xf])

    return p


@use_tempdirs
class TestLoadCase(unittest.TestCase):

    def test_load_case_unchanged_grid(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.GaussLobatto(num_segments=10))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # Initialize the system with values from the case.
        # We unnecessarily call setup again just to make sure we obliterate the previous solution
        p.setup()

        p.set_val('phase0.controls:theta', 0.0)

        # Load the values from the previous solution
        p.load_case(case)

        # Run the model to ensure we find the same output values as those that we recorded
        p.run_model()

        assert_near_equal(case.get_val('phase0.controls:theta'),
                          p.get_val('phase0.controls:theta'))

    def test_load_case_unchanged_grid_polynomial_control(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.GaussLobatto(num_segments=10), polynomial_control=True)

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # Initialize the system with values from the case.
        # We unnecessarily call setup again just to make sure we obliterate the previous solution
        p.setup()

        # Load the values from the previous solution
        p.load_case(case)

        # Run the model to ensure we find the same output values as those that we recorded
        p.run_model()

        assert_near_equal(p.get_val('phase0.controls:theta'),
                          case.get_val('phase0.controls:theta'))

    def test_load_case_lgl_to_radau(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.GaussLobatto(num_segments=10))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.Radau(num_segments=20))

        # Load the values from the previous solution
        q.load_case(case)

        # Run the model to ensure we find the same output values as those that we recorded
        q.run_driver()

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        time_val = outputs['phase0.timeseries.time']['val']
        theta_val = outputs['phase0.timeseries.theta']['val']

        time_val_uniq, idx = np.unique(time_val, return_index=True)
        theta_val_uniq = theta_val[idx]

        assert_near_equal(q['phase0.timeseries.theta'],
                          q.model.phase0.interp(xs=time_val_uniq, ys=theta_val_uniq, nodes='all'),
                          tolerance=1.0E-3)

    def test_load_case_radau_to_lgl(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.Radau(num_segments=20))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.GaussLobatto(num_segments=50))

        # Load the values from the previous solution
        q.load_case(case)

        # Run the model to ensure we find the same output values as those that we recorded
        q.run_model()

        time_p = case.get_val('phase0.timeseries.time')
        theta_p = case.get_val('phase0.timeseries.theta')

        time_q = q.get_val('phase0.timeseries.time')
        theta_q = q.get_val('phase0.timeseries.theta')

        time_p_unique, p_idx = np.unique(time_p, return_index=True)
        theta_p_unique = theta_p[p_idx]

        time_q_unique, q_idx = np.unique(time_q, return_index=True)
        theta_q_unique = theta_q[q_idx]

        assert_near_equal(q.model.phase0.interp(xs=time_p_unique, ys=theta_p_unique, nodes='all'),
                          q.model.phase0.interp(xs=time_q_unique, ys=theta_q_unique, nodes='all'),
                          tolerance=1.0E-2)

    def test_load_case_radau_to_birkhoff(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        p = setup_problem(dm.Radau(num_segments=20))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.Birkhoff(num_nodes=50))

        # Fill q with junk so that we can be sure load_case worked
        q['phase0.t_initial'] = -88
        q['phase0.t_duration'] = 88

        q['phase0.states:x'] = -88
        q['phase0.states:y'] = -88
        q['phase0.states:v'] = -88

        q['phase0.initial_states:x'] = -88
        q['phase0.initial_states:y'] = -88
        q['phase0.initial_states:v'] = -88

        q['phase0.final_states:x'] = -88
        q['phase0.final_states:y'] = -88
        q['phase0.final_states:v'] = -88

        # Load the values from the previous solution
        q.load_case(case)

        # Run the model to ensure we find the same output values as those that we recorded
        q.run_model()

        time_p = case.get_val('phase0.timeseries.time')
        theta_p = case.get_val('phase0.timeseries.theta')

        time_q = q.get_val('phase0.timeseries.time')
        theta_q = q.get_val('phase0.timeseries.theta')

        x_p = case.get_val('phase0.timeseries.x')
        y_p = case.get_val('phase0.timeseries.y')
        v_p = case.get_val('phase0.timeseries.v')

        x0_q = q.get_val('phase0.initial_states:x')
        xf_q = q.get_val('phase0.final_states:x')

        y0_q = q.get_val('phase0.initial_states:y')
        yf_q = q.get_val('phase0.final_states:y')

        v0_q = q.get_val('phase0.initial_states:v')
        vf_q = q.get_val('phase0.final_states:v')

        time_p_unique, p_idx = np.unique(time_p, return_index=True)
        theta_p_unique = theta_p[p_idx]

        time_q_unique, q_idx = np.unique(time_q, return_index=True)
        theta_q_unique = theta_q[q_idx]

        assert_near_equal(q.model.phase0.interp(xs=time_p_unique, ys=theta_p_unique, nodes='all'),
                          q.model.phase0.interp(xs=time_q_unique, ys=theta_q_unique, nodes='all'),
                          tolerance=1.0E-2)

        self.assertTrue(np.allclose(x_p[0, ...], x0_q))
        self.assertTrue(np.allclose(x_p[-1, ...], xf_q))

        self.assertTrue(np.allclose(y_p[0, ...], y0_q))
        self.assertTrue(np.allclose(y_p[-1, ...], yf_q))

        self.assertTrue(np.allclose(v_p[0, ...], v0_q))
        self.assertTrue(np.allclose(v_p[-1, ...], vf_q))

    def test_load_case_warn_fix_final_states(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_warnings
        import dymos as dm

        p = setup_problem(dm.Radau(num_segments=20))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.GaussLobatto(num_segments=50))

        msgs = []

        # Load the values from the previous solution
        for state_name in ['x', 'y']:
            msgs.append((UserWarning, f"phase0.states:{state_name} specifies 'fix_final=True'."
                                      f" If the given restart file has a different final value"
                                      f" this will overwrite the user-specified value"))

        with assert_warnings(msgs):
            q.load_case(case)

    def test_load_case_warn_fix_final_control(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_warning
        import dymos as dm
        p = setup_problem(dm.Radau(num_segments=10))

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.Radau(num_segments=10), fix_final_state=False, fix_final_control=True)

        msg = "phase0.controls:theta specifies 'fix_final=True'. If the given restart file has a" \
              " different final value this will overwrite the user-specified value"

        with assert_warning(UserWarning, msg):
            q.load_case(case)

    def test_load_case_warn_fix_final_polynomial_control(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_warning
        import dymos as dm
        p = setup_problem(dm.Radau(num_segments=10), polynomial_control=True,)

        # Solve for the optimal trajectory
        dm.run_problem(p)

        # Load the solution
        sol_file = p.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_file).get_case('final')

        # create a problem with a different transcription with a different number of variables
        q = setup_problem(dm.Radau(num_segments=10), polynomial_control=True,
                          fix_final_state=False, fix_final_control=True)

        # Load the values from the previous solution
        msg = "phase0.controls:theta specifies 'fix_final=True'. If the given restart file has a" \
              " different final value this will overwrite the user-specified value"

        with assert_warning(UserWarning, msg):
            q.load_case(case)

    def _load_bounded_case(self, units, timeseries_units, lower, upper, x0, xf, num_segments):
        """
        Record a linear state and control history and load it into a new problem.

        Return the loaded problem and the fraction of the phase duration at its state input
        nodes and at its control input nodes.
        """
        p = setup_bounded_problem(units, timeseries_units, lower, upper, x0, xf)

        dm.run_problem(p, run_driver=False, simulate=False)

        case = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')

        q = setup_bounded_problem(units, timeseries_units, lower, upper, x0, xf,
                                  num_segments=num_segments)
        q.load_case(case)

        gd = q.model.phase0.options['transcription'].grid_data
        frac = 0.5 * (gd.node_ptau + 1.0)

        return (q, frac[gd.subset_node_indices['state_input']],
                frac[gd.subset_node_indices['control_input']])

    def test_load_case_bounds_and_timeseries_units(self):
        from openmdao.utils.assert_utils import assert_near_equal

        # units, timeseries units, lower, upper, initial value, final value
        cases = [('rad', 'rad', -0.15, 0.15, 0.0, 0.05),
                 ('rad', 'deg', -0.15, 0.15, 0.0, 0.05),
                 ('rad', 'deg', 0.02, None, 0.03, 0.05),
                 ('rad', 'deg', None, 0.15, 0.03, 0.05),
                 ('K', 'degC', 280.0, 320.0, 290.0, 300.0)]

        for units, timeseries_units, lower, upper, x0, xf in cases:
            # Load into the grid of the recorded case and into a finer one.
            for num_segments in (4, 6):
                with self.subTest(units=units, timeseries_units=timeseries_units, lower=lower,
                                  upper=upper, num_segments=num_segments):
                    q, state_frac, control_frac = self._load_bounded_case(
                        units, timeseries_units, lower, upper, x0, xf, num_segments)

                    # The values are within the bounds, so they are loaded unchanged.
                    assert_near_equal(q.get_val('phase0.states:x', units=units).ravel(),
                                      x0 + (xf - x0) * state_frac, tolerance=1.0E-12)
                    assert_near_equal(q.get_val('phase0.controls:u', units=units).ravel(),
                                      x0 + (xf - x0) * control_frac, tolerance=1.0E-12)

    def test_load_case_clips_to_bounds_in_timeseries_units(self):
        from openmdao.utils.assert_utils import assert_near_equal

        x0, xf = 0.0, 0.05
        lower, upper = 0.01, 0.03

        for timeseries_units in ('rad', 'deg'):
            with self.subTest(timeseries_units=timeseries_units):
                # The bounds are relaxed when recording and tightened when loading.
                p = setup_bounded_problem('rad', timeseries_units, -1.0, 1.0, x0, xf)

                dm.run_problem(p, run_driver=False, simulate=False)

                case = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')

                q = setup_bounded_problem('rad', timeseries_units, lower, upper, x0, xf,
                                          num_segments=6)
                q.load_case(case)

                x = q.get_val('phase0.states:x', units='rad')
                u = q.get_val('phase0.controls:u', units='rad')

                # Values outside of the bounds are moved to the bounds, in the units of the bounds.
                assert_near_equal(x.min(), lower, tolerance=1.0E-12)
                assert_near_equal(x.max(), upper, tolerance=1.0E-12)
                assert_near_equal(u.min(), lower, tolerance=1.0E-12)
                assert_near_equal(u.max(), upper, tolerance=1.0E-12)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
