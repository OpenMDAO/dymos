from __future__ import print_function, absolute_import, division

import unittest


class TestBrachistochronePolynomialControl(unittest.TestCase):

    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, GaussLobatto
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9,
                                     targets='theta')

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665, targets='g')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_radau(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, Radau
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rungekutta(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, RungeKutta
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlBoundaryConstrained(unittest.TestCase):

    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, GaussLobatto
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=3, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta', loc='initial', units='deg', equals=1.0)
        phase.add_boundary_constraint('theta', loc='final', units='deg', equals=100.)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_radau(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, Radau
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta', loc='initial', units='deg', equals=1.0)
        phase.add_boundary_constraint('theta', loc='final', units='deg', equals=100.)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rungekutta(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, RungeKutta
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta', loc='initial', units='deg', equals=1.0)
        phase.add_boundary_constraint('theta', loc='final', units='deg', equals=100.)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlPathConstrained(unittest.TestCase):

    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, GaussLobatto
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=3, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta', units='deg', lower=5, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_radau(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, Radau
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta', units='deg', lower=5, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rungekutta(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, RungeKutta
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta', units='deg', lower=5, upper=120)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlRatePathConstrained(unittest.TestCase):

    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, GaussLobatto
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=3, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate', units='deg/s', lower=0, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_radau(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, Radau
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate', units='deg/s', lower=0, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rungekutta(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, RungeKutta
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate', units='deg/s', lower=0, upper=120)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlRate2PathConstrained(unittest.TestCase):

    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, GaussLobatto
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate2', units='deg/s**2', lower=-0.01, upper=0.01)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_radau(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, Radau
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate2', units='deg/s**2', lower=-0.01, upper=0.01)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    def test_brachistochrone_polynomial_control_rungekutta(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, RungeKutta
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=2, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate2', units='deg/s**2', lower=-0.01, upper=0.01)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

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

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


class TestBrachistochronePolynomialControlSimulation(unittest.TestCase):

    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, GaussLobatto
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_controls:theta')

        assert_rel_error(self, theta_exp[0], theta_imp[0])
        assert_rel_error(self, theta_exp[-1], theta_imp[-1])

    def test_brachistochrone_polynomial_control_radau(self):
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, Radau
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_controls:theta')

        assert_rel_error(self, theta_exp[0], theta_imp[0])
        assert_rel_error(self, theta_exp[-1], theta_imp[-1])

    def test_brachistochrone_polynomial_control_rungekutta(self):
        from openmdao.api import Problem, Group, DirectSolver, ScipyOptimizeDriver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase, RungeKutta
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = Phase(ode_class=BrachistochroneODE,
                      transcription=RungeKutta(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        # phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)
        phase.add_polynomial_control('theta', order=1, units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver()

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.polynomial_controls:theta'][:] = 5.0

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        theta_imp = p.get_val('phase0.timeseries.polynomial_controls:theta')
        theta_exp = exp_out.get_val('phase0.timeseries.polynomial_controls:theta')

        assert_rel_error(self, theta_exp[0], theta_imp[0])
        assert_rel_error(self, theta_exp[-1], theta_imp[-1])
