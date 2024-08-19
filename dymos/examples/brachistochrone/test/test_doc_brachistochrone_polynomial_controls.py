import unittest

try:
    import matplotlib
except ImportError:
    matplotlib = None

from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestBrachistochronePolynomialControl(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_radau(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_birkhoff(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Birkhoff(num_nodes=15))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


@use_tempdirs
class TestBrachistochronePolynomialControlBoundaryConstrained(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta', loc='initial', lower=0, upper=1.0)
        phase.add_boundary_constraint('theta', loc='final', lower=100, upper=105.)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_radau(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta', loc='initial', lower=0, upper=1.0)
        phase.add_boundary_constraint('theta', loc='final', lower=100, upper=105.)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_birkhoff(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Birkhoff(num_nodes=15))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta', loc='initial', lower=0, upper=1.0)
        phase.add_boundary_constraint('theta', loc='final', upper=105.0)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


@use_tempdirs
class TestBrachistochronePolynomialControlPathConstrained(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta', lower=0, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_radau(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta', lower=1, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_birkhoff(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Birkhoff(num_nodes=15))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta', lower=1, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


@use_tempdirs
class TestBrachistochronePolynomialControlRatePathConstrained(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate', lower=0, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_radau(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate', lower=0, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_birkhoff(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Birkhoff(num_nodes=15))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate', lower=0, upper=120)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


@use_tempdirs
class TestBrachistochronePolynomialControlRate2PathConstrained(unittest.TestCase):

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate2', lower=-0.01, upper=0.01)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_radau(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate2', lower=-0.01, upper=0.01)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_birkhoff(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Birkhoff(num_nodes=15))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('theta_rate2', lower=-0.01, upper=0.01)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.x')
        y_imp = p.get_val('phase0.timeseries.y')

        x_exp = exp_out.get_val('phase0.timeseries.x')
        y_exp = exp_out.get_val('phase0.timeseries.y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()

        t_imp = p.get_val('phase0.timeseries.time')
        theta_imp = p.get_val('phase0.timeseries.theta')

        ax.plot(t_imp, theta_imp, 'ro', label='solution')

        ax.set_xlabel('time (s)')
        ax.set_ylabel(r'$\theta$ (deg)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()


@use_tempdirs
class TestBrachistochronePolynomialControlSimulation(unittest.TestCase):

    def test_brachistochrone_polynomial_control_gauss_lobatto(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        theta_imp = p.get_val('phase0.timeseries.theta')
        theta_exp = exp_out.get_val('phase0.timeseries.theta')

        assert_near_equal(theta_exp[0], theta_imp[0])
        assert_near_equal(theta_exp[-1], theta_imp[-1])

    def test_brachistochrone_polynomial_control_radau(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        theta_imp = p.get_val('phase0.timeseries.theta')
        theta_exp = exp_out.get_val('phase0.timeseries.theta')

        assert_near_equal(theta_exp[0], theta_imp[0])
        assert_near_equal(theta_exp[-1], theta_imp[-1])

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_brachistochrone_polynomial_control_birkhoff(self):
        import matplotlib
        matplotlib.use('Agg')
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Birkhoff(num_nodes=15))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        exp_out = phase.simulate()

        theta_imp = p.get_val('phase0.timeseries.theta')
        theta_exp = exp_out.get_val('phase0.timeseries.theta')

        assert_near_equal(theta_exp[0], theta_imp[0])
        assert_near_equal(theta_exp[-1], theta_imp[-1])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
