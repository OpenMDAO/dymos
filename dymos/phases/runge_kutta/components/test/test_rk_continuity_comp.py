from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, NonlinearRunOnce, NonlinearBlockGS, \
    NewtonSolver, ExecComp, DirectSolver
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

from dymos.phases.runge_kutta.components.runge_kutta_continuity_comp import RungeKuttaContinuityComp
from dymos.phases.runge_kutta.test.rk_test_ode import TestODE


class TestRungeKuttaContinuityComp(unittest.TestCase):

    def test_continuity_comp_no_iteration(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False}}

        p = Problem(model=Group())

        p.model.add_subsystem('state_advance_comp',
                              ExecComp('y_f = y_i + 1.0',
                                       y_f={'value': np.zeros(num_seg), 'units': 'm'},
                                       y_i={'value': np.zeros(num_seg), 'units': 'm'}))

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaContinuityComp(num_segments=num_seg,
                                                       state_options=state_options),
                              promotes_outputs=['*'])

        p.model.connect('states:y', 'state_advance_comp.y_i', src_indices=[0, 1, 2, 3])
        p.model.connect('state_advance_comp.y_f', 'continuity_comp.final_states:y')

        p.model.nonlinear_solver = NonlinearRunOnce()
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000],
                                  [1.425130208333333],
                                  [2.639602661132812],
                                  [4.006818970044454],
                                  [5.301605229265987]])

        p.run_model()
        p.model.run_apply_nonlinear()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)

        expected_final = p['states:y'][:-1] + 1.0
        expected_resids = np.zeros((num_seg + 1, 1))
        expected_resids[1:, ...] = expected_final - p['states:y'][1:, ...]

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['continuity_comp']['states:y', 'final_states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'final_states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'final_states:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = -1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

    def test_continuity_comp_nonlinearblockgs(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False}}

        p = Problem(model=Group())

        p.model.add_subsystem('state_advance_comp',
                              ExecComp('y_f = y_i + 1.0',
                                       y_f={'value': np.zeros(num_seg), 'units': 'm'},
                                       y_i={'value': np.zeros(num_seg), 'units': 'm'}))

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaContinuityComp(num_segments=num_seg,
                                                       state_options=state_options),
                              promotes_outputs=['*'])

        p.model.connect('states:y', 'state_advance_comp.y_i', src_indices=[0, 1, 2, 3])
        p.model.connect('state_advance_comp.y_f', 'continuity_comp.final_states:y')

        p.model.nonlinear_solver = NonlinearBlockGS(iprint=2)
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000],
                                  [1.425130208333333],
                                  [2.639602661132812],
                                  [4.006818970044454],
                                  [5.301605229265987]])

        p.run_model()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)
        expected_resids = np.zeros((num_seg + 1, 1))

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['continuity_comp']['states:y', 'final_states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'final_states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'final_states:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = -1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

    def test_continuity_comp_newton(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False}}

        p = Problem(model=Group())

        p.model.add_subsystem('state_advance_comp',
                              ExecComp('y_f = y_i + 1.0',
                                       y_f={'value': np.zeros(num_seg), 'units': 'm'},
                                       y_i={'value': np.zeros(num_seg), 'units': 'm'}))

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaContinuityComp(num_segments=num_seg,
                                                       state_options=state_options),
                              promotes_outputs=['*'])

        p.model.connect('states:y', 'state_advance_comp.y_i', src_indices=[0, 1, 2, 3])
        p.model.connect('state_advance_comp.y_f', 'continuity_comp.final_states:y')

        p.model.nonlinear_solver = NewtonSolver(iprint=2)
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000],
                                  [1.425130208333333],
                                  [2.639602661132812],
                                  [4.006818970044454],
                                  [5.301605229265987]])

        p.run_model()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)
        expected_resids = np.zeros((num_seg + 1, 1))

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['continuity_comp']['states:y', 'final_states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'final_states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'final_states:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = -1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)
