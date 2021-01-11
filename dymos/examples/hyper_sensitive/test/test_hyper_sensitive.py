from __future__ import print_function, division, absolute_import

import os
import unittest
import warnings
from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import printoptions
from openmdao.utils.testing_utils import use_tempdirs
from dymos import Trajectory, GaussLobatto, Phase, Radau
from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE
import numpy as np
import dymos as dm

from openmdao.utils.general_utils import set_pyoptsparse_opt
_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)

tf = np.float128(10)


@use_tempdirs
class TestHyperSensitive(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def make_problem(self, transcription=GaussLobatto, optimizer='SLSQP', numseg=30):
        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = optimizer

        if optimizer == 'SNOPT':
            p.driver.declare_coloring()
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
            # p.driver.opt_settings['nlp_scaling_method'] = 'user-scaling'
            p.driver.opt_settings['print_level'] = 4
            p.driver.opt_settings['linear_solver'] = 'mumps'
            p.driver.declare_coloring()
        else:
            p.driver.declare_coloring()

        traj = p.model.add_subsystem('traj', Trajectory())
        phase0 = traj.add_phase('phase0', Phase(ode_class=HyperSensitiveODE,
                                                transcription=transcription(num_segments=numseg, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot')
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L')
        phase0.add_control('u', opt=True, targets=['u'])

        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        phase0.set_refine_options(refine=True, tol=1e-7, max_order=14)

        p.setup(check=True)

        p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[1.5, 1], nodes='state_input'))
        p.set_val('traj.phase0.states:xL', phase0.interpolate(ys=[0, 1], nodes='state_input'))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase0.interpolate(ys=[-0.6, 2.4],
                                                               nodes='control_input'))

        return p

    def solution(self):
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
        p = self.make_problem(transcription=Radau, optimizer='SLSQP')
        p.run_model()
        with printoptions(linewidth=1024, edgeitems=100):
            cpd = p.check_partials(method='fd', compact_print=True, out_stream=None)

    @unittest.skipIf(optimizer is not 'IPOPT', 'IPOPT not available')
    def test_hyper_sensitive_radau(self):
        p = self.make_problem(transcription=Radau, optimizer='IPOPT')
        dm.run_problem(p, refine_iteration_limit=5)
        ui, uf, J = self.solution()

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[0],
                          ui,
                          tolerance=5e-6)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[-1],
                          uf,
                          tolerance=5e-6)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:xL')[-1],
                          J,
                          tolerance=1e-6)

    @unittest.skipIf(optimizer is not 'IPOPT', 'IPOPT not available')
    def test_hyper_sensitive_gauss_lobatto(self):
        p = self.make_problem(transcription=GaussLobatto, optimizer='IPOPT')
        dm.run_problem(p, refine_iteration_limit=5)

        ui, uf, J = self.solution()

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[0],
                          ui,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.controls:u')[-1],
                          uf,
                          tolerance=1e-4)

        assert_near_equal(p.get_val('traj.phase0.timeseries.states:xL')[-1],
                          J,
                          tolerance=1e-4)

    @unittest.skipIf(optimizer is not 'IPOPT', 'IPOPT not available')
    def test_refinement_warning(self):
        p = self.make_problem(transcription=Radau, optimizer='IPOPT')

        msg = "Refinement not performed. Set run_driver to True to perform refinement."

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            dm.run_problem(p, run_driver=False, refine_iteration_limit=10)

        self.assertIn(msg, [str(w.message) for w in ctx])
