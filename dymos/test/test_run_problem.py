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

import openmdao
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

om_version = tuple([int(s) for s in openmdao.__version__.split('-')[0].split('.')])


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

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        dm.run_problem(p, simulate=True, simulate_kwargs={'times_per_seg': 100})

        self.assertTrue(os.path.exists('dymos_solution.db'))
        # Assert the results are what we expect.
        sol_case = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case = om.CaseReader('dymos_simulation.db').get_case('final')

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

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

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

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        dm.run_problem(p, refine_iteration_limit=20)

        self.assertTrue(os.path.exists('dymos_solution.db'))
        # Assert the results are what we expect.
        cr = om.CaseReader('dymos_solution.db')
        case = cr.get_case('final')
        assert_almost_equal(case.outputs['traj.phase0.timeseries.time'].max(), 1.8016, decimal=4)

        cr_opt = om.CaseReader('brach_driver_rec.db')
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

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        dm.run_problem(p, refine_iteration_limit=20, case_prefix='brach_test')

        self.assertTrue(os.path.exists('dymos_solution.db'))
        # Assert the results are what we expect.
        cr = om.CaseReader('dymos_solution.db')
        case = cr.get_case('brach_test_final')
        assert_almost_equal(case.outputs['traj.phase0.timeseries.time'].max(), 1.8016, decimal=4)

        cr_opt = om.CaseReader('brach_driver_rec.db')
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

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        pos0 = [0, 10]
        posf = [10, 5]

        p['phase0.states:pos'] = phase.interp('pos', [pos0, posf])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

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

    def test_restart_from_file(self):
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
        run_problem(q, run_driver=False, simulate=False, restart='vanderpol_simulation.sql')

        # s = q.model.traj.simulate(rtol=1.0E-9, atol=1.0E-9)
        s = om.CaseReader('vanderpol_simulation.sql').get_case('final')

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
        fx1s = interp1d(ts, x1s, kind='cubic')
        fx0s = interp1d(ts, x0s, kind='cubic')
        fus = interp1d(ts, us, kind='cubic')

        assert_almost_equal(x1q, fx1s(tq), decimal=2)
        assert_almost_equal(x0q, fx0s(tq), decimal=2)
        assert_almost_equal(uq, fus(tq), decimal=5)

    def test_restart_from_case(self):
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
        s = om.CaseReader('vanderpol_simulation.sql').get_case('final')
        run_problem(q, run_driver=False, simulate=False, restart=s)

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
        fx1s = interp1d(ts, x1s, kind='cubic')
        fx0s = interp1d(ts, x0s, kind='cubic')
        fus = interp1d(ts, us, kind='cubic')

        assert_almost_equal(x1q, fx1s(tq), decimal=2)
        assert_almost_equal(x0q, fx0s(tq), decimal=2)
        assert_almost_equal(uq, fus(tq), decimal=5)


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

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase0.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase0.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase0.interp('theta', [5, 100]))
        p.set_val('traj.phase0.parameters:g', 9.80665)

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
        simulation_record_file = 'simulation_record_file.db'
        dm.run_problem(self.p, simulate=True, simulation_record_file=simulation_record_file)

        self.assertTrue(os.path.exists(simulation_record_file))

    def test_run_brachistochrone_problem_set_solution_record_file(self):
        solution_record_file = 'solution_record_file.db'
        dm.run_problem(self.p, solution_record_file=solution_record_file)

        self.assertTrue(os.path.exists(solution_record_file))

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

        sol = sol_results.get_val('traj.phase0.parameter_vals:array')
        sim = sim_results.get_val('traj.phase0.parameter_vals:array')

        assert_near_equal(sol - sim, np.zeros_like(sol))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
