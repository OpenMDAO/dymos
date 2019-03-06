from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, NonlinearRunOnce, NonlinearBlockGS, \
    NewtonSolver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

from dymos.phases.runge_kutta.components.runge_kutta_state_continuity_comp import \
    RungeKuttaStateContinuityComp


class TestRungeKuttaContinuityCompScalar(unittest.TestCase):

    def test_continuity_comp_scalar_no_iteration_fwd(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'],
                               'time_direction': 'forward'}}

        p = Problem(model=Group())

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaStateContinuityComp(num_segments=num_seg,
                                                            state_options=state_options),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.model.nonlinear_solver = NonlinearRunOnce()
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000],
                                  [1.425130208333333],
                                  [2.639602661132812],
                                  [4.006818970044454],
                                  [5.301605229265987]])

        p['state_integrals:y'] = np.array([[1.0],
                                           [1.0],
                                           [1.0],
                                           [1.0]])

        p.run_model()
        p.model.run_apply_nonlinear()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)

        y_f = p['states:y'][1:, ...]
        y_i = p['states:y'][:-1, ...]
        dy_given = y_f - y_i
        dy_computed = p['state_integrals:y']

        expected_resids = np.zeros((num_seg + 1, 1))
        expected_resids[1:, ...] = dy_given - dy_computed

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = -1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

    def test_continuity_comp_scalar_nonlinearblockgs_fwd(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False, 'time_direction': 'forward'}}

        p = Problem(model=Group())

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaStateContinuityComp(num_segments=num_seg,
                                                            state_options=state_options),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.model.nonlinear_solver = NonlinearBlockGS(iprint=2)
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000],
                                  [1.425130208333333],
                                  [2.639602661132812],
                                  [4.006818970044454],
                                  [5.301605229265987]])

        p['state_integrals:y'] = np.array([[1.0],
                                           [1.0],
                                           [1.0],
                                           [1.0]])

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

        J_fwd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = -1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

    def test_continuity_comp_scalar_newton_fwd(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False, 'time_direction': 'forward'}}

        p = Problem(model=Group())

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaStateContinuityComp(num_segments=num_seg,
                                                            state_options=state_options),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.model.nonlinear_solver = NewtonSolver(iprint=2)
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000],
                                  [1.425130208333333],
                                  [2.639602661132812],
                                  [4.006818970044454],
                                  [5.301605229265987]])

        p['state_integrals:y'] = np.array([[1.0],
                                           [1.0],
                                           [1.0],
                                           [1.0]])

        p.run_model()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)
        expected_resids = np.zeros((num_seg + 1, 1))

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = -1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)


class TestRungeKuttaContinuityCompVector(unittest.TestCase):

    def test_continuity_comp_vector_no_iteration_fwd(self):
        num_seg = 2
        state_options = {'y': {'shape': (2,), 'units': 'm', 'targets': ['y'],
                               'time_direction': 'forward'}}

        p = Problem(model=Group())

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaStateContinuityComp(num_segments=num_seg,
                                                            state_options=state_options),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.model.nonlinear_solver = NonlinearRunOnce()
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000, 2.639602661132812],
                                  [1.425130208333333, 4.006818970044454],
                                  [2.639602661132812, 5.301605229265987]])

        p['state_integrals:y'] = np.array([[1.0, 1.0],
                                           [1.0, 1.0]])

        p.run_model()
        p.model.run_apply_nonlinear()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)

        y_f = p['states:y'][1:, ...]
        y_i = p['states:y'][:-1, ...]
        dy_given = y_f - y_i
        dy_computed = p['state_integrals:y']

        expected_resids = np.zeros((num_seg + 1,) + state_options['y']['shape'])
        expected_resids[1:, ...] = dy_given - dy_computed

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs')

        J_fwd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        size = np.prod(state_options['y']['shape'])
        J_fd[:size, :size] = -np.eye(size)

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

    def test_continuity_comp_vector_nonlinearblockgs_fwd(self):
        num_seg = 2
        state_options = {'y': {'shape': (2,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False, 'time_direction': 'forward'}}

        p = Problem(model=Group())

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaStateContinuityComp(num_segments=num_seg,
                                                            state_options=state_options),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.model.nonlinear_solver = NonlinearBlockGS(iprint=2)
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000, 2.639602661132812],
                                  [1.425130208333333, 4.006818970044454],
                                  [2.639602661132812, 5.301605229265987]])

        p['state_integrals:y'] = np.array([[1.0, 1.0],
                                           [1.0, 1.0]])

        p.run_model()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)
        expected_resids = np.zeros((num_seg + 1, 2))

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        size = np.prod(state_options['y']['shape'])
        J_fd[:size, :size] = -np.eye(size)

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

    def test_continuity_comp_vector_newton_fwd(self):
        num_seg = 2
        state_options = {'y': {'shape': (2,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False, 'time_direction': 'forward'}}

        p = Problem(model=Group())

        p.model.add_subsystem('continuity_comp',
                              RungeKuttaStateContinuityComp(num_segments=num_seg,
                                                            state_options=state_options),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.model.nonlinear_solver = NewtonSolver(iprint=2)
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000, 2.639602661132812],
                                  [1.425130208333333, 4.006818970044454],
                                  [2.639602661132812, 5.301605229265987]])

        p['state_integrals:y'] = np.array([[1.0, 1.0],
                                           [1.0, 1.0]])

        p.run_model()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)
        expected_resids = np.zeros((num_seg + 1, 2))

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['continuity_comp.states:y']['resids'], expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['continuity_comp']['states:y', 'states:y']['J_fd']

        size = np.prod(state_options['y']['shape'])
        J_fd[:size, :size] = -np.eye(size)

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)
