import unittest

import numpy as np

import openmdao.api as om

import dymos
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.misc import GroupWrapperConfig
from dymos.utils.testing_utils import PhaseStub, SimpleODE
from dymos.transcriptions.pseudospectral.components import BirkhoffIterGroup
from dymos.phase.options import StateOptionsDictionary, TimeOptionsDictionary
from dymos.transcriptions.grid_data import BirkhoffGrid


BirkhoffIterGroup = GroupWrapperConfig(BirkhoffIterGroup, [PhaseStub()])


@use_tempdirs
class TestBirkhoffIterGroup(unittest.TestCase):

    def test_solve_segments_gl_fwd(self):
        for grid_type in ['lgl', 'cgl']:
            with self.subTest(msg=grid_type):
                with dymos.options.temporary(include_check_partials=True):

                    state_options = {'x': StateOptionsDictionary()}

                    state_options['x']['shape'] = (1,)
                    state_options['x']['units'] = 's**2'
                    state_options['x']['targets'] = ['x']
                    state_options['x']['initial_bounds'] = (None, None)
                    state_options['x']['final_bounds'] = (None, None)
                    state_options['x']['solve_segments'] = 'forward'
                    state_options['x']['rate_source'] = 'x_dot'

                    time_options = TimeOptionsDictionary()
                    grid_data = BirkhoffGrid(num_nodes=31, grid_type=grid_type)
                    nn = grid_data.subset_num_nodes['all']
                    ode_class = SimpleODE

                    p = om.Problem()
                    p.model.add_subsystem('birkhoff', BirkhoffIterGroup(state_options=state_options,
                                                                        time_options=time_options,
                                                                        grid_data=grid_data,
                                                                        ode_class=ode_class))

                    birkhoff = p.model._get_subsystem('birkhoff')

                    birkhoff.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
                    birkhoff.linear_solver = om.DirectSolver()

                    p.setup(force_alloc_complex=True)

                    # Instead of using the TimeComp just transform the node segment taus onto [0, 2]
                    times = grid_data.node_stau + 1

                    solution = np.reshape(times**2 + 2 * times + 1 - 0.5 * np.exp(times), (nn, 1))
                    dsolution_dt = np.reshape(2 * times + 2 - 0.5 * np.exp(times), (nn, 1))

                    p.set_val('birkhoff.initial_states:x', 0.5)
                    p.set_val('birkhoff.ode_all.t', times)
                    p.set_val('birkhoff.ode_all.p', 1.0)

                    # We don't need to provide guesses for these values in this case
                    # p.set_val('birkhoff.final_states:x', solution[-1])
                    # p.set_val('birkhoff.states:x', solution)
                    # p.set_val('birkhoff.state_rates:x', dsolution_dt)

                    p.final_setup()
                    p.run_model()

                    assert_near_equal(solution, p.get_val('birkhoff.states:x'), tolerance=1.0E-9)
                    assert_near_equal(dsolution_dt, p.get_val('birkhoff.state_rates:x'), tolerance=1.0E-9)
                    assert_near_equal(solution[np.newaxis, 0], p.get_val('birkhoff.initial_states:x'), tolerance=1.0E-9)
                    assert_near_equal(solution[np.newaxis, -1], p.get_val('birkhoff.final_states:x'), tolerance=1.0E-9)

                    cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)
                    assert_check_partials(cpd)

    def test_solve_segments_gl_bkwd(self):

        for grid_type in ['lgl', 'cgl']:
            with self.subTest(msg=grid_type):
                with dymos.options.temporary(include_check_partials=True):

                    state_options = {'x': StateOptionsDictionary()}

                    state_options['x']['shape'] = (1,)
                    state_options['x']['units'] = 's**2'
                    state_options['x']['targets'] = ['x']
                    state_options['x']['initial_bounds'] = (None, None)
                    state_options['x']['final_bounds'] = (None, None)
                    state_options['x']['solve_segments'] = 'backward'
                    state_options['x']['rate_source'] = 'x_dot'

                    time_options = TimeOptionsDictionary()
                    grid_data = BirkhoffGrid(num_nodes=31, grid_type=grid_type)
                    nn = grid_data.subset_num_nodes['all']
                    ode_class = SimpleODE

                    p = om.Problem()
                    p.model.add_subsystem('birkhoff', BirkhoffIterGroup(state_options=state_options,
                                                                        time_options=time_options,
                                                                        grid_data=grid_data,
                                                                        ode_class=ode_class))

                    birkhoff = p.model._get_subsystem('birkhoff')

                    birkhoff.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
                    birkhoff.linear_solver = om.DirectSolver()

                    p.setup(force_alloc_complex=True)

                    # Instead of using the TimeComp just transform the node segment taus onto [0, 2]
                    times = grid_data.node_stau + 1

                    solution = np.reshape(times**2 + 2 * times + 1 - 0.5 * np.exp(times), (nn, 1))
                    dsolution_dt = np.reshape(2 * times + 2 - 0.5 * np.exp(times), (nn, 1))

                    p.set_val('birkhoff.final_states:x', solution[-1])
                    p.set_val('birkhoff.ode_all.t', times)
                    p.set_val('birkhoff.ode_all.p', 1.0)

                    # We don't need to provide guesses for these values in this case
                    # p.set_val('birkhoff.final_states:x', solution[-1])
                    # p.set_val('birkhoff.states:x', solution)
                    # p.set_val('birkhoff.state_rates:x', dsolution_dt)

                    p.final_setup()
                    p.run_model()

                    assert_near_equal(solution, p.get_val('birkhoff.states:x'), tolerance=1.0E-9)
                    assert_near_equal(dsolution_dt, p.get_val('birkhoff.state_rates:x'), tolerance=1.0E-9)
                    assert_near_equal(solution[np.newaxis, 0], p.get_val('birkhoff.initial_states:x'), tolerance=1.0E-9)
                    assert_near_equal(solution[np.newaxis, -1], p.get_val('birkhoff.final_states:x'), tolerance=1.0E-9)

                    cpd = p.check_partials(method='cs', compact_print=True, out_stream=None)
                    assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
