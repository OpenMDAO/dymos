import time
import unittest

import numpy as np

import openmdao.api as om

import dymos
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.misc import GroupWrapperConfig
from dymos.utils.testing_utils import PhaseStub, SimpleODE
from dymos.transcriptions.common.time_comp import TimeComp
from dymos.transcriptions.picard_shooting.multiple_shooting_iter_group import MultipleShootingIterGroup
from dymos.phase.options import StateOptionsDictionary, TimeOptionsDictionary
from dymos.transcriptions.grid_data import GaussLobattoGrid, ChebyshevGaussLobattoGrid


TimeComp = GroupWrapperConfig(TimeComp, [])
MultipleShootingIterGroupWrapped = GroupWrapperConfig(MultipleShootingIterGroup, [PhaseStub()])


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


@use_tempdirs
class TestMultipleShootingIterGroup(unittest.TestCase):

    def test_multiple_shooting_iter_group(self):
        for direction in ['forward', 'backward']:
            for grid_type in [GaussLobattoGrid, ChebyshevGaussLobattoGrid]:
                for num_seg, nodes_per_seg in [(1, 21), (3, 11)]:
                    grid_data = grid_type(nodes_per_seg=nodes_per_seg, num_segments=num_seg)
                    for nl_solver in [om.NonlinearBlockGS(use_aitken=True, maxiter=100, iprint=0),
                                      om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0)]:
                        with self.subTest(msg=f'{direction=} grid={grid_data}'):
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
                                ode_class = SimpleODE

                                p = om.Problem()

                                class PhaseStub(om.Group):

                                    def setup(self):
                                        self.state_options = state_options
                                        self.time_options = time_options

                                    def classify_var(self, var):
                                        return 'ode'

                                    def configure(self):
                                        self._get_subsystem('ms').configure_io(self)
                                        self._get_subsystem('time').configure_io()

                                p.model = PhaseStub()

                                p.model.add_subsystem('time', TimeComp(num_nodes=grid_data.num_nodes,
                                                                       node_ptau=grid_data.node_ptau,
                                                                       node_dptau_dstau=grid_data.node_dptau_dstau,
                                                                       units='s'))

                                p.model.add_subsystem('ms', MultipleShootingIterGroup(state_options=state_options,
                                                                                      time_units=time_options['units'],
                                                                                      grid_data=grid_data,
                                                                                      ode_class=ode_class,
                                                                                      ms_nonlinear_solver=nl_solver),
                                                      promotes=['ode_all*'])
                                p.model.connect('time.t', 'ode_all.t')
                                p.model.connect('time.dt_dstau', 'ms.picard_update_comp.dt_dstau')

                                ms = p.model._get_subsystem('ms')

                                ms.nonlinear_solver = nl_solver
                                ms.linear_solver = om.DirectSolver()

                                p.setup(force_alloc_complex=True)

                                p.set_val('time.t_initial', 0.0)
                                p.set_val('time.t_duration', 2.0)

                                def solution(t):
                                    return t**2 + 2 * t + 1 - 0.5 * np.exp(t)

                                def dsolution_dt(t):
                                    return 2 * t + 2 - 0.5 * np.exp(t)

                                if direction == 'forward':
                                    p.set_val('ms.seg_initial_states:x', 0.5)
                                    p.set_val('ms.initial_states:x', 0.5)
                                else:
                                    p.set_val('ms.seg_final_states:x', solution(2.0))
                                    p.set_val('ms.final_states:x', solution(2.0))

                                p.set_val('ode_all.p', 1.0)

                                p.final_setup()

                                t_start = time.perf_counter()
                                p.run_model()
                                t_end = time.perf_counter()

                                # print(f"Elapsed time: {t_end-t_start:.4f} seconds")

                                t = p.get_val('time.t')
                                x = p.get_val('ms.states:x')
                                x_dot = p.get_val('ode_all.x_dot')

                                assert_near_equal(solution(t), x.ravel(), tolerance=1.0E-9)
                                assert_near_equal(dsolution_dt(t), x_dot.ravel(), tolerance=1.0E-9)
                                assert_near_equal(solution(0), p.get_val('ms.initial_states:x').ravel(), tolerance=1.0E-9)
                                assert_near_equal(solution(t[-1]), p.get_val('ms.final_states:x').ravel(), tolerance=1.0E-9)

                                cpd = p.check_partials(method='fd', compact_print=False, out_stream=None)
                                assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)


if __name__ == '__main__':
    unittest.main()
