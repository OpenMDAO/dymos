from __future__ import print_function, division, absolute_import
import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.general_utils import set_pyoptsparse_opt
_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)

import dymos as dm
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestRunProblem(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @unittest.skipIf(optimizer is not 'IPOPT', 'IPOPT not available')
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
            p.driver.opt_settings['print_level'] = 5
            p.driver.opt_settings['max_iter'] = 200
            p.driver.opt_settings['linear_solver'] = 'mumps'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE,
                                                   transcription=dm.Radau(num_segments=10, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
        phase0.add_control('u', opt=True, targets=['u'], rate_continuity=False)

        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        phase0.set_refine_options(refine=True, tol=1e-6)

        p.setup(check=True)

        tf = np.float128(20)

        p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[1.5, 1], nodes='state_input'))
        p.set_val('traj.phase0.states:xL', phase0.interpolate(ys=[0, 1], nodes='state_input'))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase0.interpolate(ys=[-0.6, 2.4],
                                                               nodes='control_input'))
        dm.run_problem(p, refine_method='hp', refine_iteration_limit=10)

        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
                   (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[0],
                          ui,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[-1],
                          uf,
                          tolerance=5e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:xL')[-1],
                          J,
                          tolerance=5e-4)

    @unittest.skipIf(optimizer is not 'IPOPT', 'IPOPT not available')
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
            p.driver.opt_settings['print_level'] = 5
            p.driver.opt_settings['linear_solver'] = 'mumps'

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE,
                                                   transcription=dm.GaussLobatto(num_segments=20, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
        phase0.add_control('u', opt=True, targets=['u'], rate_continuity=False)

        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        phase0.set_refine_options(refine=True, tol=1.0E-5)

        p.setup(check=True)

        tf = 20

        p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[1.5, 1], nodes='state_input'))
        p.set_val('traj.phase0.states:xL', phase0.interpolate(ys=[0, 1], nodes='state_input'))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase0.interpolate(ys=[-0.6, 2.4],
                                                               nodes='control_input'))
        dm.run_problem(p, refine_method='hp', refine_iteration_limit=5)

        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
                   (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)

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
                                                   transcription=dm.Radau(num_segments=10, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                         units=BrachistochroneODE.states['x']['units'],
                         fix_initial=True, fix_final=False, solve_segments=False)
        phase0.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                         units=BrachistochroneODE.states['y']['units'],
                         fix_initial=True, fix_final=False, solve_segments=False)
        phase0.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                         units=BrachistochroneODE.states['v']['units'],
                         fix_initial=True, fix_final=False, solve_segments=False)
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

        p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[0, 10], nodes='state_input'))
        p.set_val('traj.phase0.states:y', phase0.interpolate(ys=[10, 5], nodes='state_input'))
        p.set_val('traj.phase0.states:v', phase0.interpolate(ys=[0, 9.9], nodes='state_input'))
        p.set_val('traj.phase0.controls:theta', phase0.interpolate(ys=[5, 100], nodes='control_input'))
        p.set_val('traj.phase0.parameters:g', 9.80665)

        dm.run_problem(p)

    def test_modify_problem(self):
        from dymos.examples.vanderpol.vanderpol_dymos import vanderpol
        from dymos.examples.vanderpol.vanderpol_dymos_plots import vanderpol_dymos_plots
        from dymos.run_problem import modify_problem, run_problem
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

        # Call modify_problem with simulation restart database
        # modify_problem(q, restart='vanderpol_simulation.sql')

        # # Run the model
        run_problem(q, restart='vanderpol_simulation.sql')

        #  The solution should look like the explicit time history for the states and controls.
        DO_PLOTS = False
        if DO_PLOTS:
            vanderpol_dymos_plots(q)  # only for visual inspection and debug
        else:  # automate comparison
            s = q.model.traj.simulate()

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


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
