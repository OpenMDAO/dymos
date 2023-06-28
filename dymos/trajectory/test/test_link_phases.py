import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal


class ODEComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('h', shape=nn, units='m')
        self.add_input('v', shape=nn, units='m/s')
        self.add_output('hdot', shape=nn, units='m/s')
        self.add_output('vdot', shape=nn, units='m/s**2')

        self.declare_partials(of='hdot', wrt='v', rows=np.arange(nn), cols=np.arange(nn), val=1.0)

    def compute(self, inputs, outputs):
        outputs['hdot'] = inputs['v']
        outputs['vdot'] = -9.80665


@use_tempdirs
class TestPhaseLinkageComp(unittest.TestCase):

    @staticmethod
    def make_problem(link_all_vars=False, connected=True):
        dm.options['include_check_partials'] = True
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)

        t = dm.Radau(num_segments=1, order=3, compressed=True)
        phase0 = dm.Phase(ode_class=ODEComp, transcription=t)

        traj.add_phase('phase0', phase0)
        phase0.set_time_options(fix_initial=True, fix_duration=True)

        phase0.add_state('h', fix_initial=True, fix_final=False,
                         solve_segments='forward', rate_source='hdot')
        phase0.add_state('v', fix_initial=True, fix_final=False,
                         solve_segments='forward', rate_source='vdot')

        phase1 = dm.Phase(ode_class=ODEComp, transcription=t)

        traj.add_phase('phase1', phase1)
        phase1.set_time_options(input_initial=connected, fix_duration=True)

        phase1.add_state('h', input_initial=connected, fix_final=False,
                         solve_segments='forward', rate_source='hdot')
        phase1.add_state('v', input_initial=connected, fix_final=False,
                         solve_segments='forward', rate_source='vdot')

        linked_vars = ['*'] if link_all_vars else ['time', 'h', 'v']
        traj.link_phases(['phase0', 'phase1'], vars=linked_vars, connected=connected)

        phase1.add_objective('h', loc='final')

        p.setup()

        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 1.0)
        p.set_val('traj.phase0.states:h', phase0.interp('h', [19.6133, 15]))
        p.set_val('traj.phase0.states:v', phase0.interp('v', [0, -5]))

        p.set_val('traj.phase1.t_duration', 1.0)

        return p

    def test_link_phases_specified_vars_unconnected(self):
        p = self.make_problem(connected=False)
        dm.run_problem(p, run_driver=True)

        h0 = p.get_val('traj.phase0.states:h')[-1]
        h1 = p.get_val('traj.phase1.states:h')[0]
        v0 = p.get_val('traj.phase0.states:v')[-1]
        v1 = p.get_val('traj.phase1.states:v')[0]

        assert_near_equal(h0, h1)
        assert_near_equal(v0, v1)

    def test_link_phases_all_vars_unconnected(self):
        p = self.make_problem(link_all_vars=True, connected=False)
        dm.run_problem(p, run_driver=True)

        h0 = p.get_val('traj.phase0.states:h')[-1]
        h1 = p.get_val('traj.phase1.states:h')[0]
        v0 = p.get_val('traj.phase0.states:v')[-1]
        v1 = p.get_val('traj.phase1.states:v')[0]

        assert_near_equal(h0, h1)
        assert_near_equal(v0, v1)

    def test_link_phases_specified_vars_connected(self):
        p = self.make_problem(connected=True)
        dm.run_problem(p, run_driver=False)

        h0 = p.get_val('traj.phase0.states:h')[-1]
        h1 = p.get_val('traj.phase1.states:h')[0]
        v0 = p.get_val('traj.phase0.states:v')[-1]
        v1 = p.get_val('traj.phase1.states:v')[0]

        assert_near_equal(h0, h1)
        assert_near_equal(v0, v1)

    def test_link_phases_all_vars_connected(self):
        p = self.make_problem(link_all_vars=True, connected=True)
        dm.run_problem(p, run_driver=False)

        h0 = p.get_val('traj.phase0.states:h')[-1]
        h1 = p.get_val('traj.phase1.states:h')[0]
        v0 = p.get_val('traj.phase0.states:v')[-1]
        v1 = p.get_val('traj.phase1.states:v')[0]

        assert_near_equal(h0, h1)
        assert_near_equal(v0, v1)
