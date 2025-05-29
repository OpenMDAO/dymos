import time
import unittest

import numpy as np

import openmdao.api as om

import dymos
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.misc import GroupWrapperConfig, om_version
from dymos.utils.testing_utils import PhaseStub, SimpleODE
from dymos.transcriptions.picard_shooting.birkhoff_picard_iter_group import BirkhoffPicardIterGroup
from dymos.phase.options import StateOptionsDictionary, TimeOptionsDictionary
from dymos.transcriptions.grid_data import BirkhoffGrid, GaussLobattoGrid


BirkhoffPicardIterGroup = GroupWrapperConfig(BirkhoffPicardIterGroup, [PhaseStub()])


@use_tempdirs
@unittest.skipIf(om_version()[0] < (3, 37, 0), 'Requires OpenMDAO version later than 3.37.0')
class TestBirkhoffPicardIterGroup(unittest.TestCase):

    def test_birkhoff_picard_solve_segments(self):
        for direction in ['forward', 'backward']:
            for grid_type in ['lgl', 'cgl']:
                for nl_solver in ['newton', 'nlbgs']:
                    for num_segments, nodes_per_seg in [(1, 11)]:
                        with self.subTest(msg=f'{direction=} {grid_type=} {nl_solver=}'):
                            with dymos.options.temporary(include_check_partials=True):

                                state_options = {'x': StateOptionsDictionary()}

                                state_options['x']['shape'] = (1,)
                                state_options['x']['units'] = 's**2'
                                state_options['x']['targets'] = ['x']

                                state_options['x']['initial_bounds'] = (None, None)
                                state_options['x']['final_bounds'] = (None, None)
                                state_options['x']['solve_segments'] = direction
                                state_options['x']['rate_source'] = 'x_dot'

                                time_options = TimeOptionsDictionary()
                                grid_data = GaussLobattoGrid(nodes_per_seg=nodes_per_seg, num_segments=num_segments)

                                nn = grid_data.subset_num_nodes['all']
                                ode_class = SimpleODE

                                p = om.Problem()
                                p.model.add_subsystem('birkhoff', BirkhoffPicardIterGroup(state_options=state_options,
                                                                                          time_units=time_options['units'],
                                                                                          grid_data=grid_data,
                                                                                          ode_class=ode_class))

                                birkhoff = p.model._get_subsystem('birkhoff')

                                if nl_solver == 'newton':
                                    birkhoff.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100)
                                elif nl_solver == 'nlbgs':
                                    birkhoff.nonlinear_solver = om.NonlinearBlockGS(maxiter=201, use_aitken=True)
                                else:
                                    birkhoff.nonlinear_solver = om.NonlinearRunOnce()
                                birkhoff.linear_solver = om.DirectSolver()

                                p.setup(force_alloc_complex=True)

                                # Instead of using the TimeComp just transform the node segment taus onto [0, 2]
                                times = (grid_data.node_stau + 1) * 1.0

                                solution = np.reshape(times**2 + 2 * times + 1 - 0.5 * np.exp(times), (nn, 1))
                                dsolution_dt = np.reshape(2 * times + 2 - 0.5 * np.exp(times), (nn, 1))

                                p.final_setup()

                                if direction == 'forward':
                                    p.set_val('birkhoff.picard_update_comp.seg_initial_states:x', 0.5)
                                else:
                                    p.set_val('birkhoff.picard_update_comp.seg_final_states:x', solution[-1])
                                p.set_val('birkhoff.ode_all.t', times)
                                p.set_val('birkhoff.ode_all.p', 1.0)

                                p.final_setup()

                                t_start = time.perf_counter()
                                p.run_model()
                                t_end = time.perf_counter()

                                print(f"Elapsed time: {t_end-t_start:.4f} seconds")

                                assert_near_equal(solution, p.get_val('birkhoff.states:x'), tolerance=1.0E-9)
                                assert_near_equal(dsolution_dt.ravel(),
                                                  p.get_val('birkhoff.picard_update_comp.f_computed:x').ravel(),
                                                  tolerance=1.0E-9)

                                cpd = p.check_partials(method='cs', compact_print=False,
                                                       show_only_incorrect=True, out_stream=None)
                                assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @unittest.skip('This test is a demonstation of the inability of Birkhoff-Picard '
                   'iteration to solve highly nonlinear systems.')
    def test_birkhoff_solve_segments_lorenz(self):

        class LorenzAttractorODE(om.JaxExplicitComponent):
            """
            The ODE for the Lorenz attractor.
            """

            def initialize(self):
                """
                All Dymos ODE systems are required to have an option "num_nodes",
                which is the number of points at which the ODE is simultaneously evaluated.

                This will be set by the Phase during setup once the transcription details are known.
                """
                self.options.declare('num_nodes', types=(int,))

            def setup(self):
                """
                In setup, we add inputs and outputs.

                The first dimension is assumed to pertain to the index of the node.

                An input that's a scalar at each node should have a shape of
                (num_nodes, 1) or just (num_nodes,).

                For vectors or matrices, it's just the shape of the matrix at each
                node prepended with num_nodes.

                We provide units for the scalars, but OpenMDAO doesn't do unit conversion on an index-by-index basis,
                so we just assume that no unit conversion should be done for the S matrix and K vector.
                """
                nn = self.options['num_nodes']

                # ODE inputs
                self.add_input('x', shape=(nn,))
                self.add_input('y', shape=(nn,))
                self.add_input('z', shape=(nn,))
                self.add_input('s', shape=(1,))
                self.add_input('r', shape=(1,))
                self.add_input('b', shape=(1,))

                # State rates
                self.add_output('x_dot', shape=(nn,), units='1/s', tags=['dymos.state_rate_source:x'])
                self.add_output('y_dot', shape=(nn,), units='1/s', tags=['dymos.state_rate_source:y'])
                self.add_output('z_dot', shape=(nn,), units='1/s', tags=['dymos.state_rate_source:z'])

            # because our compute primal output depends on static variables, in this case
            # and self.options['num_nodes'], we must define a get_self_statics method. This method must
            # return a tuple of all static variables. Their order in the tuple doesn't matter.  If your
            # component happens to have discrete inputs, do NOT return them here. Discrete inputs are passed
            # into the compute_primal function individually, after the continuous variables.
            def get_self_statics(self):
                # return value must be hashable
                return self.options['num_nodes'],

            def setup_partials(self):
                nn = self.options['num_nodes']
                ar = np.arange(nn, dtype=int)
                self.declare_partials('x_dot', 'x', rows=ar, cols=ar)
                self.declare_partials('x_dot', 'y', rows=ar, cols=ar)
                self.declare_partials('x_dot', 's')

                self.declare_partials('y_dot', 'x', rows=ar, cols=ar)
                self.declare_partials('y_dot', 'y', rows=ar, cols=ar)
                self.declare_partials('y_dot', 'z', rows=ar, cols=ar)
                self.declare_partials('y_dot', 'r')

                self.declare_partials('z_dot', 'x', rows=ar, cols=ar)
                self.declare_partials('z_dot', 'y', rows=ar, cols=ar)
                self.declare_partials('z_dot', 'z', rows=ar, cols=ar)
                self.declare_partials('z_dot', 'b')

            def compute_primal(self, x, y, z, s=10, r=28, b=2.667):
                """
                Parameters
                ----------
                x : array-like
                    The x state.
                y : array-like
                    The y state.
                z : array-like
                    The z state.
                s : float
                    The Lorenz s parameter.
                r : float
                    The Lorenz r parameter.
                b : float
                    The Lorenz b parameter.

                Returns
                -------
                x_dot : array-like
                    The time-derivative of the x state.
                y_dot : array-like
                    The time-derivative of the y state.
                z_dot : array-like
                    The time-derivative of the z state.
                """
                x_dot = s * (y - x)
                y_dot = r * x - y - x*z
                z_dot = x * y - b * z
                return x_dot, y_dot, z_dot

        for direction in ['forward']:
            for grid_type in ['lgl']:
                with self.subTest(msg=grid_type):
                    with dymos.options.temporary(include_check_partials=True):

                        state_options = {'x': StateOptionsDictionary(),
                                         'y': StateOptionsDictionary(),
                                         'z': StateOptionsDictionary()}

                        state_options['x']['shape'] = (1,)
                        state_options['x']['units'] = None
                        state_options['x']['targets'] = ['x']
                        state_options['x']['initial_bounds'] = (None, None)
                        state_options['x']['final_bounds'] = (None, None)
                        state_options['x']['solve_segments'] = direction
                        state_options['x']['rate_source'] = 'x_dot'

                        state_options['y']['shape'] = (1,)
                        state_options['y']['units'] = None
                        state_options['y']['targets'] = ['y']
                        state_options['y']['initial_bounds'] = (None, None)
                        state_options['y']['final_bounds'] = (None, None)
                        state_options['y']['solve_segments'] = direction
                        state_options['y']['rate_source'] = 'y_dot'

                        state_options['z']['shape'] = (1,)
                        state_options['z']['units'] = None
                        state_options['z']['targets'] = ['z']
                        state_options['z']['initial_bounds'] = (None, None)
                        state_options['z']['final_bounds'] = (None, None)
                        state_options['z']['solve_segments'] = direction
                        state_options['z']['rate_source'] = 'z_dot'

                        time_options = TimeOptionsDictionary()
                        grid_data = BirkhoffGrid(num_nodes=300, grid_type=grid_type)
                        nn = grid_data.subset_num_nodes['all']
                        ode_class = LorenzAttractorODE

                        p = om.Problem()
                        p.model.add_subsystem('birkhoff', BirkhoffPicardIterGroup(state_options=state_options,
                                                                                  time_options=time_options,
                                                                                  grid_data=grid_data,
                                                                                  ode_class=ode_class))

                        birkhoff = p.model._get_subsystem('birkhoff')

                        birkhoff.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0)
                        birkhoff.nonlinear_solver = om.NonlinearBlockGS(maxiter=500, use_aitken=True, iprint=0)
                        birkhoff.linear_solver = om.DirectSolver()

                        p.setup(force_alloc_complex=True)

                        # Instead of using the TimeComp just transform the node segment taus onto [0, 100]
                        times = (grid_data.node_stau + 1) * 0.3
                        dt_dstau = (times[-1] / 2.0) * np.ones(nn)

                        if direction == 'forward':
                            p.set_val('birkhoff.initial_states:x', 0.0)
                            p.set_val('birkhoff.initial_states:y', 1.0)
                            p.set_val('birkhoff.initial_states:z', 1.05)

                            p.set_val('birkhoff.states:x', 0.0)
                            p.set_val('birkhoff.states:y', 1.0)
                            p.set_val('birkhoff.states:z', 1.05)
                        else:
                            raise ValueError('not set up for backwards prop yet')
                            # p.set_val('birkhoff.final_states:x', solution[-1])
                        p.set_val('birkhoff.dt_dstau', dt_dstau)
                        p.set_val('birkhoff.ode_all.s', 10.0)
                        p.set_val('birkhoff.ode_all.r', 28.0)
                        p.set_val('birkhoff.ode_all.b', 2.667)

                        p.final_setup()
                        p.run_model()

                    p.model.list_vars()

                    x = p.get_val('birkhoff.states:x')
                    y = p.get_val('birkhoff.states:y')
                    z = p.get_val('birkhoff.states:z')

                    import matplotlib.pyplot as plt
                    ax = plt.figure().add_subplot(projection='3d')
                    ax.plot(x, y, z, '-')
                    plt.show()

                    # assert_near_equal(solution, p.get_val('birkhoff.states:x'), tolerance=1.0E-9)
                    # assert_near_equal(dsolution_dt.ravel(), p.get_val('birkhoff.state_rates:x'), tolerance=1.0E-9)
                    # assert_near_equal(solution[np.newaxis, 0], p.get_val('birkhoff.initial_states:x'), tolerance=1.0E-9)
                    # assert_near_equal(solution[np.newaxis, -1], p.get_val('birkhoff.final_states:x'), tolerance=1.0E-9)

                    # cpd = p.check_partials(method='fd', compact_print=True, out_stream=None)
                    # assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)


if __name__ == '__main__':
    unittest.main()
