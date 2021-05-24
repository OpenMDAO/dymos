"""
Unit tests for tagging state rate targets in a model.
"""
import unittest
import warnings

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


class BrachistochroneODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('static_gravity', types=(bool,), default=False,
                             desc='If True, treat gravity as a static (scalar) input, rather than '
                                  'having different values at each node.')

    def setup(self):
        nn = self.options['num_nodes']
        g_default_val = 9.80665 if self.options['static_gravity'] else 9.80665 * np.ones(nn)

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=g_default_val, desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.ones(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s',
                        tags=['dymos.state_rate_source:x', 'dymos.state_units:m'])

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s',
                        tags=['dymos.state_rate_source:y', 'dymos.state_units:m'])

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])

        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)

        if self.options['static_gravity']:
            c = np.zeros(self.options['num_nodes'])
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=c)
        else:
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, partials):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        partials['vdot', 'g'] = cos_theta
        partials['vdot', 'theta'] = -g * sin_theta

        partials['xdot', 'v'] = sin_theta
        partials['xdot', 'theta'] = v * cos_theta

        partials['ydot', 'v'] = -cos_theta
        partials['ydot', 'theta'] = v * sin_theta

        partials['check', 'v'] = 1 / sin_theta
        partials['check', 'theta'] = -v * cos_theta / sin_theta ** 2


class ODEGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        num_nodes = self.options['num_nodes']
        self.add_subsystem('ode', BrachistochroneODE(num_nodes=num_nodes),
                           promotes=['*'])


@use_tempdirs
class TestStateDiscovery(unittest.TestCase):

    def test_discovery(self):

        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=ODEGroup,
                                        transcription=dm.GaussLobatto(num_segments=10)))

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665)

        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        p['traj.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj.phase0.controls:theta'] = phase.interp('theta', [5, 100.5])

        dm.run_problem(p)

    def test_error_messages(self):

        class BadComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']
                self.add_input('x', np.ones((nn, )))
                self.add_output('xdot', np.ones((nn, )), tags=['dymos.state_units:x'])

        p = om.Problem()

        phase = dm.Phase(ode_class=BadComp,
                         transcription=dm.GaussLobatto(num_segments=2))
        p.model.add_subsystem('phase', phase)

        with self.assertRaises(ValueError) as cm:
            p.setup()

        msg = ("'dymos.state_units:x' tag declared on 'xdot' also requires "
               "that the 'dymos.state_rate_source:x' tag be declared.")
        self.assertEqual(str(cm.exception), msg)

        class BadComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']
                self.add_input('x', np.ones((nn, )))
                self.add_output('xdot', np.ones((nn, )), tags=['dymos.state_rate_source:x'])

        p = om.Problem()

        phase = dm.Phase(ode_class=BadComp,
                         transcription=dm.GaussLobatto(num_segments=2))
        phase.set_state_options('x', rate_source='foo')

        p.model.add_subsystem('phase', phase)

        with self.assertRaises(ValueError) as cm:
            p.setup()

        msg = ("rate_source has been declared twice for state "
               "'x' which is tagged on 'xdot'.")
        self.assertEqual(str(cm.exception), msg)

        class BadComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']
                self.add_input('x', np.ones((nn, )))
                self.add_output('xdot', np.ones((nn, )))

        p = om.Problem()

        phase = dm.Phase(ode_class=BadComp,
                         transcription=dm.GaussLobatto(num_segments=2))
        phase.add_state('x')

        p.model.add_subsystem('phase', phase)

        with self.assertRaises(ValueError) as cm:
            p.setup()

        msg = ("State 'x' is missing a rate_source.")
        self.assertEqual(str(cm.exception), msg)

    def test_deprecations(self):

        class DepComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']
                self.add_input('x', np.ones((nn, )), units='m')
                self.add_output('xdot', np.ones((nn, )), units='m/s',
                                tags=['state_rate_source:x', 'state_units:m'])

        p = om.Problem()

        phase = dm.Phase(ode_class=DepComp,
                         transcription=dm.GaussLobatto(num_segments=2))
        p.model.add_subsystem('phase', phase)

        expected_msg0 = "The tag 'state_rate_source:x' has a deprecated format and will no longer " \
                        "work in dymos version 2.0.0. Use 'dymos.state_rate_source:x' instead."

        expected_msg1 = "The tag 'state_units:m' has a deprecated format and will no longer work " \
                        "in dymos version 2.0.0. Use 'dymos.state_units:m' instead."

        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)

        self.assertIn(expected_msg0, [str(w.message) for w in ctx])
        self.assertIn(expected_msg1, [str(w.message) for w in ctx])


if __name__ == "__main__":
    unittest.main()
