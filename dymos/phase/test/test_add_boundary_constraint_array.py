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

        rs = np.arange(2*nn, dtype=int)
        cs = np.repeat(np.arange(nn, dtype=int), 2)

        self.declare_partials('*', '*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_sparsity=True)

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
class TestAddBoundaryConstraint(unittest.TestCase):

    def test_simple_no_exception(self):
        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        transcription = dm.GaussLobatto(num_segments=3,
                                        order=3,
                                        compressed=True, solve_segments='forward')
        fix_final = True

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

        # test add_boundary_constraint with arrays:
        expected = np.array([10, 5])
        phase.add_boundary_constraint(name='pos', loc='final', lower=expected-1)
        phase.add_boundary_constraint(name='pos', loc='final', upper=expected+1)
        phase.add_boundary_constraint(name='pos', loc='final', equals=expected)

        phase.add_state('v',
                        rate_source='vdot', units='m/s',
                        fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', val=9.80665, opt=False)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)
        p.final_setup()

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 1.8016

        pos0 = [0, 10]
        posf = [10, 5]

        p['traj0.phase0.states:pos'] = phase.interpolate(ys=[pos0, posf], nodes='state_input')
        p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['traj0.phase0.parameters:g'] = 9.80665

        p.run_driver()
