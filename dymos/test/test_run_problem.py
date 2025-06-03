from __future__ import print_function, division, absolute_import
import os
import unittest
import pathlib

import numpy as np
from numpy.testing import assert_almost_equal

try:
    import matplotlib
except ImportError:
    matplotlib = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.general_utils import set_pyoptsparse_opt

import dymos as dm
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.examples.brachistochrone.brachistochrone_vector_states_ode import BrachistochroneVectorStatesODE
from dymos.utils.testing_utils import _get_reports_dir

_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)


@use_tempdirs
class TestRunProblem(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    def test_run_HS_problem_radau_hp_refine(self):
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
            p.driver.opt_settings['print_level'] = 0
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

        tf = 20.0

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

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[0],
                          ui,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[-1],
                          uf,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.xL')[-1],
                          J,
                          tolerance=5e-4)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_run_HS_problem_radau_ph_refine(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = optimizer

        if optimizer == 'SNOPT':
            p.driver.opt_settings['Major iterations limit'] = 200
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['print_level'] = 0
            p.driver.opt_settings['max_iter'] = 200
            p.driver.opt_settings['linear_solver'] = 'mumps'
            p.driver.opt_settings['mu_strategy'] = 'monotone'
            p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
            p.driver.opt_settings['mu_init'] = 0.01

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

        tf = 20.0

        p.set_val('traj.phase0.states:x', phase0.interp('x', [1.5, 1]))
        p.set_val('traj.phase0.states:xL', phase0.interp('xL', [0, 1]))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase0.interp('u', [-0.6, 2.4]))
        dm.run_problem(p, refine_method='ph', refine_iteration_limit=10)

        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) *
                   np.exp(-2 * val) - (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[0],
                          ui,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[-1],
                          uf,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.xL')[-1],
                          J,
                          tolerance=5e-4)

    @require_pyoptsparse(optimizer='IPOPT')
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
            p.driver.opt_settings['print_level'] = 0
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

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[0],
                          ui,
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[-1],
                          uf,
                          tolerance=1e-2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.xL')[-1],
                          J,
                          tolerance=5e-4)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_run_brachistochrone_problem(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver(print_results=False)
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=2,
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

        phase0.set_time_val(initial=0.0, duration=2.0)
        phase0.set_state_val('x', [0, 10])
        phase0.set_state_val('y', [10, 5])
        phase0.set_state_val('v', [0, 9.9])
        phase0.set_control_val('theta', [5, 100])
        phase0.set_parameter_val('g', 9.80665)

        dm.run_problem(p, simulate=True, simulate_kwargs={'times_per_seg': 100})

        # Assert the results are what we expect.
        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')
        self.assertTrue(os.path.exists(sol_db))

        sol_t = sol_case.get_val('traj.phase0.timeseries.time')
        sim_t = sim_case.get_val('traj.phase0.timeseries.time')

        assert_almost_equal(sol_case.get_val('traj.phase0.timeseries.time')[-1, ...], 1.8016, decimal=4)
        assert_almost_equal(sim_case.get_val('traj.phase0.timeseries.time')[-1, ...], 1.8016, decimal=2)

        # With two, 3rd-order Radau segments we expect 8 points total.
        self.assertTupleEqual(sol_t.shape, (8, 1))

        # Requested 100 output times per segment, so expect 200 points total
        self.assertTupleEqual(sim_t.shape, (200, 1))

    @require_pyoptsparse(optimizer='SLSQP')
    def test_illegal_simulate_kwargs(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=2,
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

        phase0.set_time_val(initial=0.0, duration=2.0)
        phase0.set_state_val('x', [0, 10])
        phase0.set_state_val('y', [10, 5])
        phase0.set_state_val('v', [0, 9.9])
        phase0.set_control_val('theta', [5, 100])
        phase0.set_parameter_val('g', 9.80665)

        with self.assertRaises(ValueError) as e:
            dm.run_problem(p, simulate=True, simulate_kwargs={'record_file': 'my_sim_file.db'})
        self.assertEqual(str(e.exception),
                         'Key "record_file" was found in simulate_kwargs but should instead by provided by '
                         'the argument "simulation_record_file".')

        with self.assertRaises(ValueError) as e:
            dm.run_problem(p, simulate=True, simulate_kwargs={'case_prefix': 'foo'})
        self.assertEqual(str(e.exception),
                         'Key "case_prefix" was found in simulate_kwargs but should instead by provided by '
                         'the argument "case_prefix", not part of the simulate_kwargs dictionary.')

        with self.assertRaises(ValueError) as e:
            dm.run_problem(p, simulate=True, simulate_kwargs={'case_prefix': 'foo'})
        self.assertEqual(str(e.exception),
                         'Key "case_prefix" was found in simulate_kwargs but should instead by provided by '
                         'the argument "case_prefix", not part of the simulate_kwargs dictionary.')

    @require_pyoptsparse(optimizer='SLSQP')
    def test_run_brachistochrone_problem_refine_case_driver_case_prefix(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.add_recorder(om.SqliteRecorder('brach_driver_rec.db'))

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=5,
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

        phase0.set_time_val(initial=0.0, duration=2.0)
        phase0.set_state_val('x', [0, 10])
        phase0.set_state_val('y', [10, 5])
        phase0.set_state_val('v', [0, 9.9])
        phase0.set_control_val('theta', [5, 100])
        phase0.set_parameter_val('g', 9.80665)

        dm.run_problem(p, refine_iteration_limit=20)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        driver_db = p.get_outputs_dir() / 'brach_driver_rec.db'

        self.assertTrue(os.path.exists(sol_db))
        # Assert the results are what we expect.
        cr = om.CaseReader(sol_db)
        case = cr.get_case('final')
        assert_almost_equal(case.outputs['traj.phase0.timeseries.time'].max(), 1.8016, decimal=4)

        cr_opt = om.CaseReader(driver_db)
        cases = cr_opt.list_cases(source='driver', out_stream=None)

        for case in cases:
            self.assertTrue(case.startswith('hp_') and 'pyOptSparse_SLSQP|' in case, msg=f'Unexpected case: {case}')

    @require_pyoptsparse(optimizer='SLSQP')
    def test_run_brachistochrone_problem_refine_case_prefix(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.add_recorder(om.SqliteRecorder('brach_driver_rec.db'))
        p.driver.options['debug_print'] = ['desvars']

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                                   transcription=dm.Radau(num_segments=5,
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

        phase0.set_time_val(initial=0.0, duration=2.0)
        phase0.set_state_val('x', [0, 10])
        phase0.set_state_val('y', [10, 5])
        phase0.set_state_val('v', [0, 9.9])
        phase0.set_control_val('theta', [5, 100])
        phase0.set_parameter_val('g', 9.80665)

        dm.run_problem(p, refine_iteration_limit=20, case_prefix='brach_test')

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        driver_db = p.get_outputs_dir() / 'brach_driver_rec.db'

        self.assertTrue(os.path.exists(sol_db))
        # Assert the results are what we expect.
        case = om.CaseReader(sol_db).get_case('brach_test_final')
        assert_almost_equal(case.outputs['traj.phase0.timeseries.time'].max(), 1.8016, decimal=4)

        cr_opt = om.CaseReader(driver_db)
        cases = cr_opt.list_cases(source='driver', out_stream=None)

        for case in cases:
            self.assertTrue(case.startswith('brach_test_hp_') and 'pyOptSparse_SLSQP|' in case, msg=f'Unexpected case: {case}')

    @require_pyoptsparse(optimizer='SLSQP')
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

        phase.set_time_val(initial=0, duration=2.0)

        pos0 = [0, 10]
        posf = [10, 5]
        phase.set_state_val('pos', [pos0, posf])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        dm.run_problem(p, refine_iteration_limit=5)

        assert_near_equal(p.get_val('phase0.t')[-1], 1.8016, tolerance=1.0E-3)

    @require_pyoptsparse(optimizer='SLSQP')
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

        phase0.set_time_val(initial=0.0, duration=2.0)
        phase0.set_state_val('x', [0, 10])
        phase0.set_state_val('y', [10, 5])
        phase0.set_state_val('v', [0, 9.9])
        phase0.set_control_val('theta', [5, 100])
        phase0.set_parameter_val('g', 9.80665)

        dm.run_problem(p, simulate=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'

        self.assertTrue(os.path.exists(sol_db))

        # Assert the results are what we expect.
        case = om.CaseReader(sol_db).get_case('final')
        assert_almost_equal(case.outputs['traj.phase0.timeseries.time'].max(), 1.8016, decimal=4)

    def test_restart_from_file(self):
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol
        from dymos.run_problem import run_problem
        from openmdao.components.interp_util.interp import InterpND
        from numpy.testing import assert_almost_equal

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)

        # Run the problem (simulate only)
        dm.run_problem(p, run_driver=False, simulate=True)

        # Run the model
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        # create a new problem for restart to simulate a different command line execution
        q = vanderpol(transcription='gauss-lobatto', num_segments=75)

        run_problem(q, run_driver=False, simulate=False, restart=sim_db)

        sol_db2 = q.get_outputs_dir() / 'dymos_solution.db'
        s = om.CaseReader(sol_db2).get_case('final')

        # get_val returns data for duplicate time points; remove them before interpolating
        tq = q.get_val('traj.phase0.timeseries.time')[:, 0]
        nodup = np.insert(tq[1:] != tq[:-1], 0, True)
        tq = tq[nodup]
        x1q = q.get_val('traj.phase0.timeseries.x1')[:, 0][nodup]
        x0q = q.get_val('traj.phase0.timeseries.x0')[:, 0][nodup]
        uq = q.get_val('traj.phase0.timeseries.u')[:, 0][nodup]

        ts = s.get_val('traj.phase0.timeseries.time')[:, 0]
        nodup = np.insert(ts[1:] != ts[:-1], 0, True)
        ts = ts[nodup]
        x1s = s.get_val('traj.phase0.timeseries.x1')[:, 0][nodup]
        x0s = s.get_val('traj.phase0.timeseries.x0')[:, 0][nodup]
        us = s.get_val('traj.phase0.timeseries.u')[:, 0][nodup]

        # create interpolation functions so that values can be looked up at matching time points
        fx1s = InterpND('cubic', ts, x1s).interpolate
        fx0s = InterpND('cubic', ts, x0s).interpolate
        fus = InterpND('cubic', ts, us).interpolate

        assert_almost_equal(x1q, fx1s(tq), decimal=2)
        assert_almost_equal(x0q, fx0s(tq), decimal=2)
        assert_almost_equal(uq, fus(tq), decimal=5)

    def test_restart_from_case(self):
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol
        from dymos.run_problem import run_problem
        from openmdao.components.interp_util.interp import InterpND
        from numpy.testing import assert_almost_equal

        # Create the Dymos problem instance
        p = vanderpol(transcription='gauss-lobatto', num_segments=75)

        # Run the problem (simulate only)
        dm.run_problem(p, run_driver=False, simulate=True)

        # create a new problem for restart to simulate a different command line execution
        q = vanderpol(transcription='gauss-lobatto', num_segments=75)

        # Run the model
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        run_problem(q, run_driver=False, simulate=False, restart=sim_db)

        sol_db2 = q.get_outputs_dir() / 'dymos_solution.db'
        s = om.CaseReader(sol_db2).get_case('final')

        # get_val returns data for duplicate time points; remove them before interpolating
        tq = q.get_val('traj.phase0.timeseries.time')[:, 0]
        nodup = np.insert(tq[1:] != tq[:-1], 0, True)
        tq = tq[nodup]
        x1q = q.get_val('traj.phase0.timeseries.x1')[:, 0][nodup]
        x0q = q.get_val('traj.phase0.timeseries.x0')[:, 0][nodup]
        uq = q.get_val('traj.phase0.timeseries.u')[:, 0][nodup]

        ts = s.get_val('traj.phase0.timeseries.time')[:, 0]
        nodup = np.insert(ts[1:] != ts[:-1], 0, True)
        ts = ts[nodup]
        x1s = s.get_val('traj.phase0.timeseries.x1')[:, 0][nodup]
        x0s = s.get_val('traj.phase0.timeseries.x0')[:, 0][nodup]
        us = s.get_val('traj.phase0.timeseries.u')[:, 0][nodup]

        # create interpolation functions so that values can be looked up at matching time points
        fx1s = InterpND('cubic', ts, x1s).interpolate
        fx0s = InterpND('cubic', ts, x0s).interpolate
        fus = InterpND('cubic', ts, us).interpolate

        assert_almost_equal(x1q, fx1s(tq), decimal=2)
        assert_almost_equal(x0q, fx0s(tq), decimal=2)
        assert_almost_equal(uq, fus(tq), decimal=5)

    def test_simulate_support_model_options(self):

        class MyODE(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)
                self.options.declare('enable', default=False)

            def setup(self):
                nn = self.options['num_nodes']

                if not self.options['enable']:
                    raise RuntimeError("Model options not passed.")

                self.add_input('state', shape=(nn,))
                self.add_output('out', shape=(nn,))
                self.add_output('state_deriv', shape=(nn,))

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                pass

        prob = om.Problem()

        phase = dm.Phase(ode_class=MyODE,
                         transcription=dm.GaussLobatto(num_segments=3))

        phase.add_state('state', rate_source='state_deriv')
        phase.add_objective('out', loc='final', ref=1000.0)

        traj = dm.Trajectory()
        prob.model.add_subsystem('traj', traj)
        traj.add_phase('phase', phase)

        prob.model_options['*'] = {'enable': True}
        prob.setup()

        # Will raise an error during simulation if model options are not set.
        dm.run_problem(prob, make_plots=False, simulate=True)


@use_tempdirs
@require_pyoptsparse(optimizer='SLSQP')
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

        phase0.timeseries_options['include_state_rates'] = True
        phase0.timeseries_options['include_control_rates'] = True
        phase0.timeseries_options['include_t_phase'] = True

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        phase0.set_time_val(initial=0.0, duration=2.0)
        phase0.set_state_val('x', [0, 10])
        phase0.set_state_val('y', [10, 5])
        phase0.set_state_val('v', [0, 9.9])
        phase0.set_control_val('theta', [5, 100])
        phase0.set_parameter_val('g', 9.80665)

        self.p = p

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_run_brachistochrone_problem_make_plots(self):
        plots_cache = dm.options['plots']
        dm.options['plots'] = 'matplotlib'
        dm.run_problem(self.p, make_plots=True)
        plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath('plots')

        for varname in ['time_phase', 'x', 'xdot', 'y',
                        'ydot', 'v', 'vdot', 'theta', 'theta_rate', 'theta_rate2']:
            self.assertTrue(plot_dir.joinpath(f'{varname}.png').exists(),
                            msg=f'{varname}.png' + ' does not exist.')
        dm.options['plots'] = plots_cache

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_run_brachistochrone_problem_make_plots_set_plot_dir(self):
        _cache = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        dm.run_problem(self.p, make_plots=True, plot_dir="test_plot_dir")

        plot_dir = pathlib.Path(_get_reports_dir(self.p))
        for varname in ['x', 'y', 'v', 'theta']:
            plotfile = plot_dir.joinpath('test_plot_dir', f'{varname}.png')
            self.assertTrue(str(plotfile) + ' does not exist.')

        dm.options['plots'] = _cache

    def test_run_brachistochrone_problem_do_not_make_plots(self):
        dm.run_problem(self.p, make_plots=False)
        plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath('plots')

        for varname in ['time_phase', 'x', 'xdot', 'y', 'ydot', 'v',
                        'vdot', 'theta', 'theta_rate']:
            plotfile = plot_dir.joinpath(f'{varname}.png')
            self.assertFalse(plotfile.exists(), msg=f'Unexpectedly found plot file {plotfile}')

    def test_run_brachistochrone_problem_set_simulation_record_file(self):
        sim_db = 'simulation_record_file.db'
        dm.run_problem(self.p, simulate=True, simulation_record_file=sim_db)

        sim_db = self.p.model.traj.sim_prob.get_outputs_dir() / sim_db

        self.assertTrue(os.path.exists(sim_db))

    def test_run_brachistochrone_problem_set_solution_record_file(self):
        sol_db = 'solution_record_file.db'
        dm.run_problem(self.p, solution_record_file=sol_db)

        sol_db = self.p.get_outputs_dir() / sol_db

        self.assertTrue(os.path.exists(sol_db))

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_run_brachistochrone_problem_plot_simulation(self):
        plots_cache = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        dm.run_problem(self.p, make_plots=True, simulate=True)
        plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath('plots')

        for varname in ['time_phase', 'x', 'xdot', 'y',
                        'ydot', 'v', 'vdot', 'theta', 'theta_rate', 'theta_rate2']:
            plotfile = plot_dir.joinpath(f'{varname}.png')
            self.assertTrue(plotfile.exists(), msg=f'plot file {plotfile} does not exist!')
        dm.options['plots'] = plots_cache

    @unittest.skipIf(matplotlib is None, "This test requires matplotlib")
    def test_run_brachistochrone_problem_plot_no_simulation_record_file_given(self):
        plots_cache = dm.options['plots']
        dm.options['plots'] = 'matplotlib'

        dm.run_problem(self.p, make_plots=True, simulate=True)
        plot_dir = pathlib.Path(_get_reports_dir(self.p)).joinpath('plots')

        for varname in ['time_phase', 'x', 'xdot', 'y',
                        'ydot', 'v', 'vdot', 'theta', 'theta_rate', 'theta_rate2']:
            plotfile = plot_dir.joinpath(f'{varname.replace(":", "_")}.png')
            self.assertTrue(plotfile.exists(), msg=f'plot file {plotfile} does not exist!')

        dm.options['plots'] = plots_cache


@use_tempdirs
class TestSimulateArrayParam(unittest.TestCase):

    def test_simulate_array_param(self):
        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        # dummy array of data
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('array', np.linspace(1, 10, 10), units=None)

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

        # add dummy array as a parameter and connect it
        phase.add_parameter('array', units=None, shape=(10,), static_target=True)
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
        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, simulate=True)

        # Test the results
        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol_results = om.CaseReader(sol_db).get_case('final')
        sim_results = om.CaseReader(sim_db).get_case('final')

        sol = sol_results.get_val('traj.phase0.parameter_vals:array')
        sim = sim_results.get_val('traj.phase0.parameter_vals:array')

        assert_near_equal(sol - sim, np.zeros_like(sol))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
