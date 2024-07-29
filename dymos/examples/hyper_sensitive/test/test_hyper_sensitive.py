from __future__ import print_function, division, absolute_import

import os
import unittest
import warnings

import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.general_utils import printoptions
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE


@use_tempdirs
class TestHyperSensitive(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @require_pyoptsparse(optimizer='SLSQP')
    def make_problem(self, transcription=dm.GaussLobatto, optimizer='SLSQP', numseg=30, order=3,
                     solve_segments=False, tf=10):

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = optimizer

        if optimizer == 'SNOPT':
            p.driver.declare_coloring()
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Major iterations limit'] = 500
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['print_level'] = 0
            p.driver.opt_settings['mu_strategy'] = 'adaptive'
            p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
            p.driver.opt_settings['mu_init'] = 0.01
            p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
            p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
            p.driver.opt_settings['tol'] = 1.0E-6
            p.driver.declare_coloring()
        else:
            p.driver.declare_coloring()

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=numseg, order=3)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=numseg, order=3)
        elif transcription == 'birkhoff':
            t = dm.Birkhoff(num_nodes=order + 1, grid_type='lgl',
                            solve_segments=solve_segments)

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE, transcription=t))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=False, fix_final=False, rate_source='x_dot', ref=100)
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', ref=100)
        phase0.add_control('u', opt=True, targets=['u'])

        phase0.add_boundary_constraint('x', loc='initial', equals=1.5)
        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        phase0.set_refine_options(refine=True, tol=1e-7, max_order=14)
        phase0.set_simulate_options(times_per_seg=100)

        p.set_solver_print(0)
        p.setup(check=True, force_alloc_complex=True)

        phase0.set_time_val(initial=0, duration=tf)
        phase0.set_state_val('x', [1.5, 1])
        phase0.set_state_val('xL', [0, 1])
        phase0.set_control_val('u', [-0.6, 2.4])

        return p

    @staticmethod
    def solution(tf):
        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
                   (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)
        return ui, uf, J

    def test_partials(self):
        p = self.make_problem(transcription='radau-ps', optimizer='SLSQP')
        p.run_model()
        with printoptions(linewidth=1024, edgeitems=100):
            cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)
            assert_check_partials(cpd)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_hyper_sensitive_radau(self):
        tf = 10
        p = self.make_problem(transcription='radau-ps', optimizer='IPOPT', tf=tf)
        dm.run_problem(p, refine_iteration_limit=5)
        ui, uf, J = self.solution(tf)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[0],
                          ui,
                          tolerance=5e-6)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[-1],
                          uf,
                          tolerance=5e-6)

        assert_near_equal(p.get_val('traj.phase0.timeseries.xL')[-1],
                          J,
                          tolerance=1e-6)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_hyper_sensitive_birkhoff(self):
        tf = 10
        p = self.make_problem(transcription='birkhoff', optimizer='IPOPT', order=51, tf=tf)

        dm.run_problem(p, simulate=True, make_plots=True)
        ui, uf, J = self.solution(tf=tf)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[0],
                          ui,
                          tolerance=1e-3)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[-1],
                          uf,
                          tolerance=1e-3)

        assert_near_equal(p.get_val('traj.phase0.timeseries.xL')[-1],
                          J,
                          tolerance=1e-6)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_hyper_sensitive_gauss_lobatto(self):
        tf = 10
        p = self.make_problem(transcription='gauss-lobatto', optimizer='IPOPT', tf=tf)
        dm.run_problem(p, refine_iteration_limit=5,)

        ui, uf, J = self.solution(tf=tf)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[0],
                          ui,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.u')[-1],
                          uf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.xL')[-1],
                          J,
                          tolerance=1e-4)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_refinement_warning(self):
        p = self.make_problem(transcription='radau-ps', optimizer='IPOPT')

        msg = "Refinement not performed. Set run_driver to True to perform refinement."

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            dm.run_problem(p, run_driver=False, refine_iteration_limit=10)

        self.assertIn(msg, [str(w.message) for w in ctx])
