import unittest

import numpy as np

import openmdao.api as om

import dymos
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.examples.brachistochrone.brachistochrone_vector_states_ode import BrachistochroneVectorStatesODE
from dymos.utils.misc import GroupWrapperConfig
from dymos.transcriptions.pseudospectral.components.radau_iter_group import RadauIterGroup
from dymos.phase.options import StateOptionsDictionary, TimeOptionsDictionary, ControlOptionsDictionary
from dymos.transcriptions.grid_data import RadauGrid
from dymos.utils.testing_utils import PhaseStub, SimpleODE, SimpleVectorizedODE


RadauIterGroup = GroupWrapperConfig(RadauIterGroup, [PhaseStub()])


@use_tempdirs
class TestRadauIterGroup(unittest.TestCase):

    def test_solve_segments(self):
        with dymos.options.temporary(include_check_partials=True):
            for direction in ['forward', 'backward']:
                for compressed in [True, False]:
                    with self.subTest(msg=f'{direction=} {compressed=}'):

                        state_options = {'x': StateOptionsDictionary()}

                        state_options['x']['shape'] = (1,)
                        state_options['x']['units'] = 's**2'
                        state_options['x']['targets'] = ['x']
                        state_options['x']['initial_bounds'] = (None, None)
                        state_options['x']['final_bounds'] = (None, None)
                        state_options['x']['solve_segments'] = direction
                        state_options['x']['rate_source'] = 'x_dot'

                        time_options = TimeOptionsDictionary()
                        grid_data = RadauGrid(num_segments=2, nodes_per_seg=8, compressed=compressed)
                        nn = grid_data.subset_num_nodes['all']
                        ode_class = SimpleODE

                        p = om.Problem()
                        p.model.add_subsystem('radau', RadauIterGroup(state_options=state_options,
                                                                      time_options=time_options,
                                                                      grid_data=grid_data,
                                                                      ode_class=ode_class))

                        radau = p.model._get_subsystem('radau')

                        radau.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=-1)
                        radau.linear_solver = om.DirectSolver()

                        p.setup(force_alloc_complex=True)

                        # Instead of using the TimeComp just transform the node segment taus onto [0, 2]
                        times = grid_data.node_ptau + 1

                        solution = np.reshape(times**2 + 2 * times + 1 - 0.5 * np.exp(times), (nn, 1))

                        # Each segment is of the same length, so dt_dstau is constant.
                        # dt_dstau is (tf - t0) / 2.0 / num_seg
                        p.set_val('radau.dt_dstau', (times[-1] / 2.0 / grid_data.num_segments))

                        if direction == 'forward':
                            p.set_val('radau.initial_states:x', 0.5)
                        else:
                            p.set_val('radau.final_states:x', solution[-1])

                        p.set_val('radau.states:x', 0.0)
                        p.set_val('radau.ode_all.t', times)
                        p.set_val('radau.ode_all.p', 1.0)

                        p.run_model()

                        x = p.get_val('radau.states:x')
                        x_0 = p.get_val('radau.initial_states:x')
                        x_f = p.get_val('radau.final_states:x')

                        idxs = grid_data.subset_node_indices['state_input']
                        assert_near_equal(solution[idxs], x, tolerance=1.0E-5)
                        assert_near_equal(solution[np.newaxis, 0], x_0, tolerance=1.0E-7)
                        assert_near_equal(solution[np.newaxis, -1], x_f, tolerance=1.0E-7)

                        cpd = p.check_partials(method='cs', compact_print=False, out_stream=None)
                        assert_check_partials(cpd)

    def test_solve_segments_vector_states(self):
        with dymos.options.temporary(include_check_partials=True):
            for direction in ['forward', 'backward']:
                with self.subTest(msg=f'{direction=}'):

                    state_options = {'z': StateOptionsDictionary()}

                    state_options['z']['shape'] = (2,)
                    state_options['z']['units'] = 's**2'
                    state_options['z']['targets'] = ['z']
                    state_options['z']['initial_bounds'] = (None, None)
                    state_options['z']['final_bounds'] = (None, None)
                    state_options['z']['solve_segments'] = direction
                    state_options['z']['rate_source'] = 'z_dot'

                    time_options = TimeOptionsDictionary()
                    grid_data = RadauGrid(num_segments=5, nodes_per_seg=4, compressed=False)
                    nn = grid_data.subset_num_nodes['all']
                    ode_class = SimpleVectorizedODE

                    p = om.Problem()
                    p.model.add_subsystem('radau', RadauIterGroup(state_options=state_options,
                                                                  time_options=time_options,
                                                                  grid_data=grid_data,
                                                                  ode_class=ode_class))

                    radau = p.model._get_subsystem('radau')

                    radau.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100,
                                                             iprint=-1, atol=1.0E-10, rtol=1.0E-10)
                    radau.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
                    radau.linear_solver = om.DirectSolver()

                    p.setup(force_alloc_complex=True)

                    # Instead of using the TimeComp just transform the node segment taus onto [0, 2]
                    times = grid_data.node_ptau + 1

                    solution = np.reshape(times**2 + 2 * times + 1 - 0.5 * np.exp(times), (nn, 1))

                    # Each segment is of the same length, so dt_dstau is constant.
                    # dt_dstau is (tf - t0) / 2.0 / num_seg
                    p.set_val('radau.dt_dstau', (times[-1] / 2.0 / grid_data.num_segments))

                    if direction == 'forward':
                        p.set_val('radau.initial_states:z', [0.5, 0.0])
                    else:
                        p.set_val('radau.final_states:z', solution[-1], indices=om.slicer[:, 0])
                        p.set_val('radau.final_states:z', 20.0, indices=om.slicer[:, 1])

                    p.set_val('radau.states:z', 0.0)
                    p.set_val('radau.ode_all.t', times)
                    p.set_val('radau.ode_all.p', 1.0)

                    p.run_model()

                    x = p.get_val('radau.states:z')
                    x_0 = p.get_val('radau.initial_states:z')
                    x_f = p.get_val('radau.final_states:z')

                    idxs = grid_data.subset_node_indices['state_input']
                    assert_near_equal(solution[idxs], x[np.newaxis, :, 0].T, tolerance=1.0E-5)
                    assert_near_equal(solution[0], x_0[:, 0], tolerance=1.0E-5)
                    assert_near_equal(solution[-1], x_f[:, 0], tolerance=1.0E-5)

                    with np.printoptions(linewidth=1024, edgeitems=1024):
                        cpd = p.check_partials(method='cs', compact_print=False, show_only_incorrect=True, out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0)

    def test_autoivc_no_solve(self):
        with dymos.options.temporary(include_check_partials=True):
            for direction in ['forward', 'backward']:
                for compressed in [True, False]:
                    with self.subTest(msg=f'{direction=} {compressed=}'):

                        state_options = {'x': StateOptionsDictionary()}

                        state_options['x']['shape'] = (1,)
                        state_options['x']['units'] = 's**2'
                        state_options['x']['targets'] = ['x']
                        state_options['x']['initial_bounds'] = (None, None)
                        state_options['x']['final_bounds'] = (None, None)
                        state_options['x']['solve_segments'] = False
                        state_options['x']['rate_source'] = 'x_dot'

                        time_options = TimeOptionsDictionary()
                        grid_data = RadauGrid(num_segments=2, nodes_per_seg=8, compressed=compressed)
                        nn = grid_data.subset_num_nodes['all']
                        ode_class = SimpleODE

                        p = om.Problem()
                        p.model.add_subsystem('radau', RadauIterGroup(state_options=state_options,
                                                                      time_options=time_options,
                                                                      grid_data=grid_data,
                                                                      ode_class=ode_class))

                        radau = p.model._get_subsystem('radau')

                        p.setup(force_alloc_complex=True)

                        # Instead of using the TimeComp just transform the node segment taus onto [0, 2]
                        times = grid_data.node_ptau + 1

                        solution = np.reshape(times**2 + 2 * times + 1 - 0.5 * np.exp(times), (nn, 1))
                        seg_end_idxs = grid_data.subset_node_indices['segment_ends'][1::2][:-1]
                        solution_input_nodes = np.delete(solution, seg_end_idxs)

                        # Each segment is of the same length, so dt_dstau is constant.
                        # dt_dstau is (tf - t0) / 2.0 / num_seg
                        p.set_val('radau.dt_dstau', (times[-1] / 2.0 / grid_data.num_segments))

                        p.set_val('radau.initial_states:x', 0.5)
                        p.set_val('radau.final_states:x', solution[-1])

                        if compressed:
                            # Solution is definted at all nodes, need to only provide the input node values.
                            p.set_val('radau.states:x', solution_input_nodes)
                        else:
                            p.set_val('radau.states:x', solution)

                        p.set_val('radau.ode_all.t', times)
                        p.set_val('radau.ode_all.p', 1.0)

                        p.run_model()

                        x = p.get_val('radau.states:x')
                        x_0 = p.get_val('radau.initial_states:x')
                        x_f = p.get_val('radau.final_states:x')

                        idxs = grid_data.subset_node_indices['state_input']
                        assert_near_equal(solution[idxs], x, tolerance=1.0E-5)
                        assert_near_equal(solution[np.newaxis, 0], x_0, tolerance=1.0E-7)
                        assert_near_equal(solution[np.newaxis, -1], x_f, tolerance=1.0E-7)

                        cpd = p.check_partials(method='cs', compact_print=False, out_stream=None)
                        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
