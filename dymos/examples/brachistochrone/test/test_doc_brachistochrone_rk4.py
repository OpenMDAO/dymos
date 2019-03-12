from __future__ import print_function, absolute_import, division

import unittest


class TestBrachistochroneRK4Example(unittest.TestCase):

    def test_brachistochrone_for_docs_forward_shooting(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('runge-kutta',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      method='rk4')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0,
                          rate_continuity=True, rate2_continuity=True)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = DirectSolver()

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
        assert_rel_error(self, p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_for_docs_backward_shooting(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('runge-kutta',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      method='rk4')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(-2.0, -0.5))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=False)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0,
                          rate_continuity=True, rate2_continuity=True)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=0)
        phase.add_boundary_constraint('y', loc='final', equals=10)
        phase.add_boundary_constraint('v', loc='final', equals=0)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver()

        p.setup(check=True)

        p['phase0.t_initial'] = 1.8016
        p['phase0.t_duration'] = -1.8016

        p['phase0.states:x'] = 10
        p['phase0.states:y'] = 5
        p['phase0.states:v'] = 10
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p['phase0.time'][-1], -1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_for_docs_forward_shooting_path_constrained_state(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('runge-kutta',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      method='rk4')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0,
                          rate_continuity=True, rate2_continuity=True)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('y', lower=5)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = DirectSolver()

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
        assert_rel_error(self, p['phase0.time'][-1], 1.805, tolerance=1.0E-2)

        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_for_docs_forward_shooting_path_constrained_control(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('runge-kutta',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      method='rk4',
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0,
                          rate_continuity=True, rate2_continuity=True)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('theta', lower=0.01, upper=110, units='deg')

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = DirectSolver()

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
        assert_rel_error(self, p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_for_docs_forward_shooting_path_constrained_control_rate(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('runge-kutta',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      method='rk4')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0,
                          rate_continuity=True, rate2_continuity=True)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in RungeKuttaPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('theta_rate', lower=-60, upper=60, units='deg/s')

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = DirectSolver()

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
        assert_rel_error(self, p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_for_docs_forward_shooting_path_constrained_ode_output(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('runge-kutta',
                      ode_class=BrachistochroneODE,
                      num_segments=20,
                      method='rk4')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(0.5, 2.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', lower=0.01, upper=179.9, ref0=0, ref=180.0,
                          rate_continuity=True, rate2_continuity=True)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Final state values can't be controlled with simple bounds in ExplicitPhase,
        # so use nonlinear boundary constraints instead.
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('check', lower=-500, upper=500)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=1)

        p.model.linear_solver = DirectSolver()

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
        assert_rel_error(self, p['phase0.time'][-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50), record=False)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()
