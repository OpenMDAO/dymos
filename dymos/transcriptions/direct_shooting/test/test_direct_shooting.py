import unittest
import warnings

import numpy as np

import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class Simple2StateODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s**2')
        self.add_input('y', shape=(nn,), units='s')
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('p', shape=(nn,), units='s**2')

        self.add_output('x_dot', shape=(nn,), units='s')
        self.add_output('y_dot', shape=(nn,), units=None)

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='x_dot', wrt='p', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='y_dot', wrt='y', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='y_dot', wrt='t', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        t = inputs['t']
        p = inputs['p']
        outputs['x_dot'] = x - t**2 + p
        outputs['y_dot'] = -2 * y + t**3 * np.exp(-2*t)

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['x_dot', 't'] = -2*t
        partials['y_dot', 't'] = -np.exp(-2*t) * t**2 * (2 * t - 3)


class Simple1StateODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('y', shape=(nn,), units=None)
        self.add_input('t', shape=(nn,), units='s')

        self.add_output('y_dot', shape=(nn,), units='1/s')

        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='y_dot', wrt='y', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='y_dot', wrt='t', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        y = inputs['y']
        t = inputs['t']
        outputs['y_dot'] = -2 * y + t**3 * np.exp(-2*t)

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['y_dot', 't'] = -np.exp(-2*t) * t**2 * (2 * t - 3)


@use_tempdirs
class TestDirectShooting(unittest.TestCase):

    def test_1_state_run_model(self):
        prob = om.Problem()

        input_grid = dm.GaussLobattoGrid(num_segments=1, nodes_per_seg=3)
        output_grid = dm.UniformGrid(num_segments=1, nodes_per_seg=11)

        tx = dm.transcriptions.DirectShooting(input_grid=input_grid,
                                              output_grid=output_grid)

        phase = dm.Phase(ode_class=Simple1StateODE, transcription=tx)

        phase.set_time_options(targets=['t'], units='s')

        # automatically discover states
        phase.set_state_options('y', targets=['y'], rate_source='y_dot')

        prob.model.add_subsystem('phase0', phase)

        prob.setup(force_alloc_complex=False)

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 1.0)
        prob.set_val('phase0.states:y', 1.0)

        prob.run_model()

        with np.printoptions(linewidth=1024):
            cpd = prob.check_partials(method='fd', compact_print=True)

        assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    def test_2_states_run_model(self):

        prob = om.Problem()

        input_grid = dm.GaussLobattoGrid(num_segments=1, nodes_per_seg=3)
        output_grid = dm.UniformGrid(num_segments=1, nodes_per_seg=11)

        tx = dm.transcriptions.DirectShooting(input_grid=input_grid,
                                              output_grid=output_grid)

        phase = dm.Phase(ode_class=Simple2StateODE, transcription=tx)

        phase.set_time_options(targets=['t'], units='s')

        # automatically discover states
        phase.set_state_options('x', targets=['x'], rate_source='x_dot')
        phase.set_state_options('y', targets=['y'], rate_source='y_dot')

        phase.add_parameter('p', targets=['p'])

        prob.model.add_subsystem('phase0', phase)

        prob.setup(force_alloc_complex=True)

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 1.0)
        prob.set_val('phase0.states:x', 0.5)
        prob.set_val('phase0.states:y', 1.0)
        prob.set_val('phase0.parameters:p', 1)

        prob.run_model()

        t_f = prob.get_val('phase0.integrator.t_final')
        x_f = prob.get_val('phase0.integrator.states_out:x')
        y_f = prob.get_val('phase0.integrator.states_out:y')


        assert_near_equal(t_f, 1.0)
        assert_near_equal(x_f[-1, ...], 2.64085909, tolerance=1.0E-5)
        assert_near_equal(y_f[-1, ...], 0.1691691, tolerance=1.0E-5)

        with np.printoptions(linewidth=1024):
            cpd = prob.check_partials(compact_print=True, method='fd')
            assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)
