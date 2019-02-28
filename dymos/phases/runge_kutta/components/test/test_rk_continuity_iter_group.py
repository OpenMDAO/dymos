from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, NonlinearRunOnce, NonlinearBlockGS, \
    NewtonSolver, ExecComp, DirectSolver
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

from dymos.phases.runge_kutta.components import RungeKuttaStateContinuityIterGroup, \
    RungeKuttaStepsizeComp
from dymos.phases.runge_kutta.test.rk_test_ode import TestODE


class TestRungeKuttaContinuityIterGroup(unittest.TestCase):

    def test_continuity_comp_no_iteration(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False, 'propagation': 'forward'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('time', val=np.array([0.00, 0.25, 0.25, 0.50,
                                             0.50, 0.75, 0.75, 1.00,
                                             1.00, 1.25, 1.25, 1.50,
                                             1.50, 1.75, 1.75, 2.00]), units='s')

        ivc.add_output('h', val=np.array([0.5, 0.5, 0.5, 0.5]), units='s')

        p.model.add_subsystem('cnty_iter_group',
                              RungeKuttaStateContinuityIterGroup(
                                  num_segments=num_seg,
                                  method='rk4',
                                  state_options=state_options,
                                  time_units='s',
                                  ode_class=TestODE,
                                  ode_init_kwargs={},
                                  k_solver_class=NonlinearRunOnce,
                                  continuity_solver_class=NonlinearRunOnce),
                              promotes_outputs=['states:*'])

        p.model.connect('h', 'cnty_iter_group.h')
        p.model.connect('time', 'cnty_iter_group.ode.t')

        src_idxs = np.arange(16, dtype=int).reshape((num_seg, 4, 1))
        p.model.connect('cnty_iter_group.ode.ydot', 'cnty_iter_group.k_comp.f:y',
                        src_indices=src_idxs, flat_src_indices=True)

        p.model.nonlinear_solver = NonlinearRunOnce()
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['states:y'] = np.array([[0.50000000],
                                  [1.425130208333333],
                                  [2.639602661132812],
                                  [4.006818970044454],
                                  [5.301605229265987]])

        p['cnty_iter_group.k_comp.k:y'] = np.array([[[0.75000000],
                                                     [0.90625000],
                                                     [0.94531250],
                                                     [1.09765625]],

                                                    [[1.087565104166667],
                                                     [1.203206380208333],
                                                     [1.232116699218750],
                                                     [1.328623453776042]],

                                                    [[1.319801330566406],
                                                     [1.368501663208008],
                                                     [1.380676746368408],
                                                     [1.385139703750610]],

                                                    [[1.378409485022227],
                                                     [1.316761856277783],
                                                     [1.301349949091673],
                                                     [1.154084459568063]]])

        p.run_model()
        p.model.run_apply_nonlinear()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)

        expected_resids = np.zeros((num_seg + 1, 1))

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['cnty_iter_group.continuity_comp.states:y']['resids'],
                         expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['cnty_iter_group.continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['cnty_iter_group.continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['cnty_iter_group.continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['cnty_iter_group.continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['cnty_iter_group.continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['cnty_iter_group.continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = 1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

    def test_continuity_comp_newtonsolver(self):
        num_seg = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y'], 'fix_initial': True,
                               'fix_final': False, 'propagation': 'forward'}}

        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('time', val=np.array([0.00, 0.25, 0.25, 0.50,
                                             0.50, 0.75, 0.75, 1.00,
                                             1.00, 1.25, 1.25, 1.50,
                                             1.50, 1.75, 1.75, 2.00]), units='s')

        ivc.add_output('h', val=np.array([0.5, 0.5, 0.5, 0.5]), units='s')

        p.model.add_subsystem('cnty_iter_group',
                              RungeKuttaStateContinuityIterGroup(
                                  num_segments=num_seg,
                                  method='rk4',
                                  state_options=state_options,
                                  time_units='s',
                                  ode_class=TestODE,
                                  ode_init_kwargs={},
                                  k_solver_class=None,
                                  continuity_solver_class=NewtonSolver),
                              promotes_outputs=['states:*'])

        p.model.connect('h', 'cnty_iter_group.h')
        p.model.connect('time', 'cnty_iter_group.ode.t')

        src_idxs = np.arange(16, dtype=int).reshape((num_seg, 4, 1))
        p.model.connect('cnty_iter_group.ode.ydot', 'cnty_iter_group.k_comp.f:y',
                        src_indices=src_idxs, flat_src_indices=True)

        p.model.nonlinear_solver = NonlinearRunOnce()
        p.model.linear_solver = DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        # Set the initial value (and all other values) of y
        p['states:y'][...] = 0.50

        p.run_model()
        # p.model.run_apply_nonlinear()

        # Test that the residuals of the states are the expected values
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=None)

        expected_resids = np.zeros((num_seg + 1, 1))

        op_dict = dict([op for op in outputs])
        assert_rel_error(self, op_dict['cnty_iter_group.continuity_comp.states:y']['resids'],
                         expected_resids)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)

        J_fwd = cpd['cnty_iter_group.continuity_comp']['states:y', 'state_integrals:y']['J_fwd']
        J_rev = cpd['cnty_iter_group.continuity_comp']['states:y', 'state_integrals:y']['J_rev']
        J_fd = cpd['cnty_iter_group.continuity_comp']['states:y', 'state_integrals:y']['J_fd']
        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)

        J_fwd = cpd['cnty_iter_group.continuity_comp']['states:y', 'states:y']['J_fwd']
        J_rev = cpd['cnty_iter_group.continuity_comp']['states:y', 'states:y']['J_rev']
        J_fd = cpd['cnty_iter_group.continuity_comp']['states:y', 'states:y']['J_fd']

        J_fd[0, 0] = 1.0

        assert_rel_error(self, J_fwd, J_rev)
        assert_rel_error(self, J_fwd, J_fd)
