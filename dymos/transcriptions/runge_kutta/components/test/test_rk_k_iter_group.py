import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

from dymos.transcriptions.runge_kutta.components.runge_kutta_k_iter_group import RungeKuttaKIterGroup
from dymos.transcriptions.runge_kutta.test.rk_test_ode import TestODE

# Modify class so we can run it standalone.
from dymos.utils.misc import GroupWrapperConfig
RungeKuttaKIterGroup = GroupWrapperConfig(RungeKuttaKIterGroup)


class TestRungeKuttaKIterGroup(unittest.TestCase):

    def test_rk4_scalar_no_iteration(self):
        num_seg = 4
        num_stages = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y']}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('initial_states_per_seg:y', shape=(num_seg, 1), units='m')
        ivc.add_output('h', shape=(num_seg, 1), units='s')
        ivc.add_output('t', shape=(num_seg * num_stages, 1), units='s')

        p.model.add_subsystem('k_iter_group',
                              RungeKuttaKIterGroup(num_segments=num_seg,
                                                   method='RK4',
                                                   state_options=state_options,
                                                   time_units='s',
                                                   ode_class=TestODE,
                                                   ode_init_kwargs={},
                                                   solver_class=om.NonlinearRunOnce))

        p.model.connect('t', 'k_iter_group.ode.t')
        p.model.connect('h', 'k_iter_group.h')
        p.model.connect('initial_states_per_seg:y', 'k_iter_group.initial_states_per_seg:y')

        src_idxs = np.arange(16, dtype=int).reshape((num_seg, num_stages, 1))
        p.model.connect('k_iter_group.ode.ydot', 'k_iter_group.k_comp.f:y',
                        src_indices=src_idxs, flat_src_indices=True)

        p.setup(check=True, force_alloc_complex=True)

        p['t'] = np.array([[0.00, 0.25, 0.25, 0.50,
                            0.50, 0.75, 0.75, 1.00,
                            1.00, 1.25, 1.25, 1.50,
                            1.50, 1.75, 1.75, 2.00]]).T

        p['h'] = np.array([[0.5, 0.5, 0.5, 0.5]]).T

        p['initial_states_per_seg:y'] = np.array([[0.50000000],
                                                  [1.425130208333333],
                                                  [2.639602661132812],
                                                  [4.006818970044454]])

        p['k_iter_group.k_comp.k:y'] = np.array([[[0.75000000],
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

        # Test that the residuals of k are zero (we started k at the expected converged value)
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=False)
        op_dict = dict([op for op in outputs])
        assert_almost_equal(op_dict['k_iter_group.k_comp.k:y']['resids'], 0.0)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk4_scalar_nonlinearblockgs(self):
        num_seg = 4
        num_stages = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y']}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('initial_states_per_seg:y', shape=(num_seg, 1), units='m')
        ivc.add_output('h', shape=(num_seg, 1), units='s')
        ivc.add_output('t', shape=(num_seg * num_stages, 1), units='s')

        p.model.add_subsystem('k_iter_group',
                              RungeKuttaKIterGroup(num_segments=num_seg,
                                                   method='RK4',
                                                   state_options=state_options,
                                                   time_units='s',
                                                   ode_class=TestODE,
                                                   ode_init_kwargs={},
                                                   solver_class=om.NonlinearBlockGS,
                                                   solver_options={'iprint': 2}))

        p.model.connect('t', 'k_iter_group.ode.t')
        p.model.connect('h', 'k_iter_group.h')
        p.model.connect('initial_states_per_seg:y', 'k_iter_group.initial_states_per_seg:y')

        src_idxs = np.arange(16, dtype=int).reshape((num_seg, num_stages, 1))
        p.model.connect('k_iter_group.ode.ydot', 'k_iter_group.k_comp.f:y',
                        src_indices=src_idxs, flat_src_indices=True)

        p.setup(check=True, force_alloc_complex=True)

        p['t'] = np.array([[0.00, 0.25, 0.25, 0.50,
                            0.50, 0.75, 0.75, 1.00,
                            1.00, 1.25, 1.25, 1.50,
                            1.50, 1.75, 1.75, 2.00]]).T

        p['h'] = np.array([[0.5, 0.5, 0.5, 0.5]]).T

        p['initial_states_per_seg:y'] = np.array([[0.50000000],
                                                  [1.425130208333333],
                                                  [2.639602661132812],
                                                  [4.006818970044454]])

        # Start k with a terrible guess
        p['k_iter_group.k_comp.k:y'][...] = 0

        p.run_model()

        # Test that the residuals of k are zero (we started k at the expected converged value)
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=False)
        op_dict = dict([op for op in outputs])
        assert_almost_equal(op_dict['k_iter_group.k_comp.k:y']['resids'], 0.0)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk4_scalar_newton(self):
        num_seg = 4
        num_stages = 4
        state_options = {'y': {'shape': (1,), 'units': 'm', 'targets': ['y']}}

        p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('initial_states_per_seg:y', shape=(num_seg, 1), units='m')
        ivc.add_output('h', shape=(num_seg, 1), units='s')
        ivc.add_output('t', shape=(num_seg * num_stages, 1), units='s')

        p.model.add_subsystem('k_iter_group',
                              RungeKuttaKIterGroup(num_segments=num_seg,
                                                   method='RK4',
                                                   state_options=state_options,
                                                   time_units='s',
                                                   ode_class=TestODE,
                                                   ode_init_kwargs={},
                                                   solver_class=om.NewtonSolver,
                                                   solver_options={'iprint': 2,
                                                                   'solve_subsystems': True}))

        p.model.connect('t', 'k_iter_group.ode.t')
        p.model.connect('h', 'k_iter_group.h')
        p.model.connect('initial_states_per_seg:y', 'k_iter_group.initial_states_per_seg:y')

        src_idxs = np.arange(16, dtype=int).reshape((num_seg, num_stages, 1))
        p.model.connect('k_iter_group.ode.ydot', 'k_iter_group.k_comp.f:y',
                        src_indices=src_idxs, flat_src_indices=True)

        p.setup(check=True, force_alloc_complex=True)

        p['t'] = np.array([[0.00, 0.25, 0.25, 0.50,
                            0.50, 0.75, 0.75, 1.00,
                            1.00, 1.25, 1.25, 1.50,
                            1.50, 1.75, 1.75, 2.00]]).T

        p['h'] = np.array([[0.5, 0.5, 0.5, 0.5]]).T

        p['initial_states_per_seg:y'] = np.array([[0.50000000],
                                                  [1.425130208333333],
                                                  [2.639602661132812],
                                                  [4.006818970044454]])

        # Start k with a terrible guess
        p['k_iter_group.k_comp.k:y'][...] = 0

        p.run_model()

        # Test that the residuals of k are zero (we started k at the expected converged value)
        outputs = p.model.list_outputs(print_arrays=True, residuals=True, out_stream=False)
        op_dict = dict([op for op in outputs])
        assert_almost_equal(op_dict['k_iter_group.k_comp.k:y']['resids'], 0.0)

        # Test the partials
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
