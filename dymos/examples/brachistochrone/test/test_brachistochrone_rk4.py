import unittest
import warnings


class TestBrachistochroneRK4Example(unittest.TestCase):

    def test_brachistochrone_forward_shooting(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            phase = dm.Phase(ode_class=BrachistochroneODE,
                             transcription=dm.RungeKutta(num_segments=20))
            assert issubclass(w[-1].category, DeprecationWarning)
            expected_msg = 'The RungeKutta transcription is deprecated and ' \
                           'will be removed in Dymos v1.0.0.\nFor equivalent behavior, users ' \
                           'should switch to GaussLobatto(order=3, solve_segments=True)'

            self.assertEqual(str(w[-1].message), expected_msg)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_backward_shooting(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(-2.0, -0.5))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=False, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=0)
        phase.add_boundary_constraint('y', loc='final', equals=10)
        phase.add_boundary_constraint('v', loc='final', equals=0)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=-10)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 1.8016
        p['phase0.t_duration'] = -1.8016

        p['phase0.states:x'] = 10
        p['phase0.states:y'] = 5
        p['phase0.states:v'] = 10
        p['phase0.controls:theta'] = phase.interpolate(ys=[100.5, 5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], -1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 0,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 10,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_path_constrained_state(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('y', lower=5)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.805, tolerance=1.0E-2)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_path_constrained_control(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('theta', lower=0.01, upper=110, units='deg')

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_path_constrained_control_rate(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in RungeKuttaPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('theta_rate', lower=-60, upper=60, units='deg/s')

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_path_constrained_ode_output(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('check', lower=-500, upper=500, shape=(1,), units='m/s')

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_boundary_constrained_control_rate(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in RungeKuttaPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0, units='deg/s**2')

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)
        assert_near_equal(p.get_val('phase0.timeseries.control_rates:theta_rate2')[-1, 0], 0,
                          tolerance=1.0E-6)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_boundary_constrained_parameter(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', units='deg', order=2, lower=0.01, upper=179.9)
        #
        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in RungeKuttaPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.polynomial_controls:theta'][:, 0] = [0.01, 50, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        assert_near_equal(p.get_val('phase0.timeseries.polynomial_control_rates:theta_rate2')[-1, 0],
                          0.0,
                          tolerance=1.0E-9)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_boundary_constrained_ode_output(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_polynomial_control('theta', units='deg', order=1, lower=0.01, upper=179.9)
        #
        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in RungeKuttaPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_boundary_constraint('check', loc='final', lower=-500, upper=500,
                                      shape=(1,), units='m/s')

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.polynomial_controls:theta'][:, 0] = [0.01, 100]

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)

    def test_brachistochrone_forward_shooting_path_constrained_time(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.RungeKutta(num_segments=20))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        phase.add_path_constraint('time', lower=0.0, upper=2.0)
        phase.add_path_constraint('time_phase', lower=0.0, upper=2.0)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = 0
        p['phase0.states:y'] = 10
        p['phase0.states:v'] = 0
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        assert_near_equal(exp_out.get_val('phase0.timeseries.states:x')[-1, 0], 10,
                          tolerance=1.0E-3)
        assert_near_equal(exp_out.get_val('phase0.timeseries.states:y')[-1, 0], 5,
                          tolerance=1.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
