import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


class BrachistochroneVectorStatesODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
        self.add_input('g', val=9.80665 * np.ones(nn), desc='grav. acceleration', units='m/s/s')
        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')

        self.add_output('pos_dot', val=np.zeros((nn, 2)), desc='velocity components', units='m/s')
        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')
        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant', units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials('*', '*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_sparsity=False)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['pos_dot'][:, 0] = v * sin_theta
        outputs['pos_dot'][:, 1] = -v * cos_theta

        outputs['check'] = v / sin_theta


@use_tempdirs
class TestAddPathConstraint(unittest.TestCase):

    def test_invalid_expression(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        # can't fix final position if you're solving the segments
        phase.add_state('pos',
                        rate_source='pos_dot', units='m',
                        fix_initial=True)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        phase.add_path_constraint(name='pos**2', lower=np.array([10, 5]))

        # test add_boundary_constraint with arrays:
        expected = "Unable to find the source 'pos**2' in the ODE."
        with self.assertRaises(ValueError) as e:
            p.setup()

        self.assertEqual(expected, str(e.exception))

    def test_duplicate_name(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        # can't fix final position if you're solving the segments
        phase.add_state('pos',
                        rate_source='pos_dot', units='m',
                        fix_initial=True)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        phase.add_path_constraint(name='pos', equals=np.array([10, 5]))

        # test add_boundary_constraint with arrays:
        with self.assertRaises(ValueError) as e:
            phase.add_path_constraint(name='pos=v**2', equals=np.array([10, 5]))

        expected = 'Cannot add new path constraint named `pos` and indices None.' \
                   ' The name `pos` is already in use as a path constraint'
        self.assertEqual(expected, str(e.exception))

    def test_duplicate_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=transcription)
        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj0', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        # can't fix final position if you're solving the segments
        phase.add_state('pos',
                        rate_source='pos_dot', units='m',
                        fix_initial=True)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        phase.add_path_constraint(name='pos', equals=np.array([10, 5]))

        # test add_boundary_constraint with arrays:
        with self.assertRaises(ValueError) as e:
            phase.add_path_constraint(name='pos', equals=np.array([10, 5]))

        expected = 'Cannot add new path constraint for variable `pos` and indices None. One already exists.'
        self.assertEqual(expected, str(e.exception))
