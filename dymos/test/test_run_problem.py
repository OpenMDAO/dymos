from __future__ import print_function, division, absolute_import
import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.examples.brachistochrone.brachistochrone_vector_states_ode import BrachistochroneVectorStatesODE
from openmdao.utils.general_utils import set_pyoptsparse_opt
_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)


@use_tempdirs
class TestRunProblem(unittest.TestCase):

    @unittest.skipIf(optimizer != 'IPOPT', 'IPOPT not available')
    def test_run_HS_problem_radau(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = optimizer

        if optimizer == 'SNOPT':
            p.driver.opt_settings['Major iterations limit'] = 200
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
            # p.driver.opt_settings['nlp_scaling_method'] = 'user-scaling'
            p.driver.opt_settings['print_level'] = 4
            p.driver.opt_settings['max_iter'] = 200
            p.driver.opt_settings['linear_solver'] = 'mumps'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE,
                                                   transcription=dm.Radau(num_segments=10,
                                                                          order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
        phase0.add_control('u', opt=True, targets=['u'], rate_continuity=False)

        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        phase0.set_refine_options(refine=True, tol=1e-6)

        p.setup(check=True)

        tf = np.float128(20)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [1.5, 1]))
        p.set_val('traj.phase0.states:xL', phase0.interp('xL', [0, 1]))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase0.interp('u', [-0.6, 2.4]))
        dm.run_problem(p, refine_method='hp', refine_iteration_limit=10)

        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) *
                   np.exp(-2 * val) - (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[0],
                          ui,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[-1],
                          uf,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:xL')[-1],
                          J,
                          tolerance=5e-4)

    @unittest.skipIf(optimizer != 'IPOPT', 'IPOPT not available')
    def test_run_HS_problem_gl(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = optimizer

        if optimizer == 'SNOPT':
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
            # p.driver.opt_settings['nlp_scaling_method'] = 'user-scaling'
            p.driver.opt_settings['print_level'] = 4
            p.driver.opt_settings['linear_solver'] = 'mumps'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE,
                                                   transcription=dm.GaussLobatto(num_segments=20,
                                                                                 order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
        phase0.add_control('u', opt=True, targets=['u'], rate_continuity=False)

        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        phase0.set_refine_options(refine=True, tol=1.0E-5)

        p.setup(check=True)

        tf = 20

        p.set_val('traj.phase0.states:x', phase0.interp('x', [1.5, 1]))
        p.set_val('traj.phase0.states:xL', phase0.interp('xL', [0, 1]))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase0.interp('u', [-0.6, 2.4]))
        dm.run_problem(p, refine_method='hp', refine_iteration_limit=5)

        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) *
                   np.exp(-2 * val) - (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[0],
                          ui,
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[-1],
                          uf,
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:xL')[-1],
                          J,
                          tolerance=5e-4)

    def test_run_brachistochrone_problem(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=10,
                                                                          order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=False)
        phase0.add_state('x', fix_initial=True, fix_final=False)
        phase0.add_state('y', fix_initial=True, fix_final=False)
        phase0.add_state('v', fix_initial=True, fix_final=False)
        phase0.add_control('theta', continuity=True, rate_continuity=True,
                           units='deg', lower=0.01, upper=179.9)
        phase0.add_parameter('g', units='m/s**2', val=9.80665)

        phase0.add_boundary_constraint('x', loc='final', equals=10)
        phase0.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase0.add_objective('time_phase', loc='final', scaler=10)

        phase0.set_refine_options(refine=True)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        dm.run_problem(p)

        self.assertTrue(os.path.exists('dymos_solution.db'))
        # Assert the results are what we expect.
        cr = om.CaseReader('dymos_solution.db')
        case = cr.get_case('final')
        assert_almost_equal(case.outputs['traj.phase0.timeseries.time'].max(), 1.8016, decimal=4)

    def test_run_brachistochrone_vector_states_problem(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=1, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=[True, False])
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('pos', loc='final', equals=5, indices=[1])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        pos0 = [0, 10]
        posf = [10, 5]

        p['phase0.states:pos'] = phase.interp('pos', [pos0, posf])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        dm.run_problem(p, refine_iteration_limit=5)

        assert_near_equal(p.get_val('phase0.time')[-1], 1.8016, tolerance=1.0E-3)

    def test_run_brachistochrone_problem_with_simulate(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=10,
                                                                          order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=False)
        phase0.add_state('x', fix_initial=True, fix_final=False)
        phase0.add_state('y', fix_initial=True, fix_final=False)
        phase0.add_state('v', fix_initial=True, fix_final=False)
        phase0.add_control('theta', continuity=True, rate_continuity=True,
                           units='deg', lower=0.01, upper=179.9)
        phase0.add_parameter('g', units='m/s**2', val=9.80665)

        phase0.add_boundary_constraint('x', loc='final', equals=10)
        phase0.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase0.add_objective('time_phase', loc='final', scaler=10)

        phase0.set_refine_options(refine=True)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        dm.run_problem(p, simulate=True)

        self.assertTrue(os.path.exists('dymos_solution.db'))
        # Assert the results are what we expect.
        cr = om.CaseReader('dymos_solution.db')
        case = cr.get_case('final')
        assert_almost_equal(case.outputs['traj.phase0.timeseries.time'].max(), 1.8016, decimal=4)

    def test_modify_problem(self):
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol
        from dymos.run_problem import run_problem
        from scipy.interpolate import interp1d
        from numpy.testing import assert_almost_equal

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)

        # Run the problem (simulate only)
        p.run_model()

        # simulate and record
        p.model.traj.simulate(record_file='vanderpol_simulation.sql')

        # create a new problem for restart to simulate a different command line execution
        q = vanderpol(transcription='gauss-lobatto', num_segments=75)

        # # Run the model
        run_problem(q, restart='vanderpol_simulation.sql')

        s = q.model.traj.simulate(rtol=1.0E-9, atol=1.0E-9)

        # get_val returns data for duplicate time points; remove them before interpolating
        tq = q.get_val('traj.phase0.timeseries.time')[:, 0]
        nodup = np.insert(tq[1:] != tq[:-1], 0, True)
        tq = tq[nodup]
        x1q = q.get_val('traj.phase0.timeseries.states:x1')[:, 0][nodup]
        x0q = q.get_val('traj.phase0.timeseries.states:x0')[:, 0][nodup]
        uq = q.get_val('traj.phase0.timeseries.controls:u')[:, 0][nodup]

        ts = s.get_val('traj.phase0.timeseries.time')[:, 0]
        nodup = np.insert(ts[1:] != ts[:-1], 0, True)
        ts = ts[nodup]
        x1s = s.get_val('traj.phase0.timeseries.states:x1')[:, 0][nodup]
        x0s = s.get_val('traj.phase0.timeseries.states:x0')[:, 0][nodup]
        us = s.get_val('traj.phase0.timeseries.controls:u')[:, 0][nodup]

        # create interpolation functions so that values can be looked up at matching time points
        fx1s = interp1d(ts, x1s, kind='cubic')
        fx0s = interp1d(ts, x0s, kind='cubic')
        fus = interp1d(ts, us, kind='cubic')

        assert_almost_equal(x1q, fx1s(tq), decimal=2)
        assert_almost_equal(x0q, fx0s(tq), decimal=2)
        assert_almost_equal(uq, fus(tq), decimal=5)


@use_tempdirs
class TestRunProblemPlotting(unittest.TestCase):
    def setUp(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=10,
                                                                          order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False)
        phase0.add_state('y', fix_initial=True, fix_final=False)
        phase0.add_state('v', fix_initial=True, fix_final=False)
        phase0.add_control('theta', continuity=True, rate_continuity=True,
                           units='deg', lower=0.01, upper=179.9)
        phase0.add_parameter('g', units='m/s**2', val=9.80665)

        phase0.add_boundary_constraint('x', loc='final', equals=10)
        phase0.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase0.add_objective('time_phase', loc='final', scaler=10)

        phase0.set_refine_options(refine=True)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        self.p = p

    def test_run_brachistochrone_problem_make_plots(self):
        dm.run_problem(self.p, make_plots=True)

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertTrue(os.path.exists(f'plots/{varname.replace(":","_")}.png'))

    def test_run_brachistochrone_problem_make_plots_set_plot_dir(self):
        plot_dir = "test_plot_dir"
        dm.run_problem(self.p, make_plots=True, plot_dir=plot_dir)

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertTrue(os.path.exists(f'test_plot_dir/{varname.replace(":","_")}.png'))

    def test_run_brachistochrone_problem_do_not_make_plots(self):
        dm.run_problem(self.p, make_plots=False)

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertFalse(os.path.exists(f'plots/{varname.replace(":","_")}.png'))

    def test_run_brachistochrone_problem_set_simulation_record_file(self):
        simulation_record_file = 'simulation_record_file.db'
        dm.run_problem(self.p, simulate=True, simulation_record_file=simulation_record_file)

        self.assertTrue(os.path.exists(simulation_record_file))

    def test_run_brachistochrone_problem_set_solution_record_file(self):
        solution_record_file = 'solution_record_file.db'
        dm.run_problem(self.p, solution_record_file=solution_record_file)

        self.assertTrue(os.path.exists(solution_record_file))

    def test_run_brachistochrone_problem_plot_simulation(self):
        dm.run_problem(self.p, make_plots=True, simulate=True)

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertTrue(os.path.exists(f'plots/{varname.replace(":","_")}.png'))

    def test_run_brachistochrone_problem_plot_no_simulation_record_file_given(self):
        dm.run_problem(self.p, make_plots=True, simulate=True)

        for varname in ['time_phase', 'states:x', 'state_rates:x', 'states:y',
                        'state_rates:y', 'states:v',
                        'state_rates:v', 'controls:theta', 'control_rates:theta_rate',
                        'control_rates:theta_rate2', 'parameters:g']:
            self.assertTrue(os.path.exists(f'plots/{varname.replace(":","_")}.png'))


@use_tempdirs
class TestSimulateArrayParam(unittest.TestCase):

    def test_simulate_array_param(self):
        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        transcription=dm.GaussLobatto(num_segments=10)))

        #
        # Set the variables
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='xdot')

        phase.add_state('y', fix_initial=True, fix_final=True, rate_source='ydot')

        phase.add_state('v', fix_initial=True, fix_final=False, rate_source='vdot')

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)
        phase.add_parameter('array', units=None, shape=(10,), static_target=True)

        # dummy array of data
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('array', np.linspace(1, 10, 10), units=None)
        # add dummy array as a parameter and connect it
        p.model.connect('array', 'traj.phase0.parameters:array')

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        #
        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj.phase0.controls:theta'] = phase.interp('theta', [5, 100.5])

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, simulate=True)

        # Test the results
        sol_results = om.CaseReader('dymos_solution.db').get_case('final')
        sim_results = om.CaseReader('dymos_solution.db').get_case('final')

        sol = sol_results.get_val('traj.phase0.timeseries.parameters:array')
        sim = sim_results.get_val('traj.phase0.timeseries.parameters:array')

        assert_near_equal(sol - sim, np.zeros_like(sol))

        # Test that the parameter is available in the solution and simulation files
        sol = sol_results.get_val('traj.phase0.parameters:array')
        sim = sim_results.get_val('traj.phase0.parameters:array')

        assert_near_equal(sol - sim, np.zeros_like(sol))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
