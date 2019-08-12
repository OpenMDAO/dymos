import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
from dymos.utils.lgl import lgl
from dymos.models.eom import FlightPathEOM2D

import numpy as np


class TestInputParameterConnections(unittest.TestCase):

    def test_dynamic_input_parameter_connections_radau(self):

        @dm.declare_time(units='s')
        @dm.declare_state('v', rate_source='eom.v_dot', units='m/s')
        @dm.declare_state('h', rate_source='eom.h_dot', units='m')
        @dm.declare_parameter('m', targets='sum.m', units='kg', shape=(2, 2))
        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'value': np.zeros((nn, 2, 2))},
                                                      m_tot={'value': np.zeros(nn)}))

                self.add_subsystem('eom', FlightPathEOM2D(num_nodes=nn))

                self.connect('sum.m_tot', 'eom.m')

        optimizer = 'SLSQP'
        num_segments = 1
        transcription_order = 5

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring()

        seg_ends, _ = lgl(num_segments + 1)

        phase = dm.Phase(ode_class=TrajectoryODE,
                         transcription=dm.Radau(num_segments=num_segments, order=transcription_order,
                                                segment_ends=seg_ends))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0.0, 100.0), duration_bounds=(0., 100.))

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s')

        phase.add_input_parameter('m', val=[[1, 2], [3, 4]], units='kg')

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interpolate(ys=[20, 0], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, -5], nodes='state_input')

        p.run_model()

        expected = np.broadcast_to(np.array([[1, 2], [3, 4]]),
                                   (p.model.phase0.options['transcription'].grid_data.num_nodes, 2, 2))
        assert_rel_error(self, p.get_val('phase0.rhs_all.sum.m'), expected)

    def test_static_input_parameter_connections_radau(self):

        @dm.declare_time(units='s')
        @dm.declare_state('v', rate_source='eom.v_dot', units='m/s')
        @dm.declare_state('h', rate_source='eom.h_dot', units='m')
        @dm.declare_parameter('m', targets='sum.m', units='kg', shape=(2, 2), dynamic=False)
        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'value': np.zeros((2, 2))},
                                                      m_tot={'value': np.zeros(nn)}))

                self.add_subsystem('eom', FlightPathEOM2D(num_nodes=nn))

                self.connect('sum.m_tot', 'eom.m')

        optimizer = 'SLSQP'
        num_segments = 1
        transcription_order = 5

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring()

        seg_ends, _ = lgl(num_segments + 1)

        phase = dm.Phase(ode_class=TrajectoryODE,
                         transcription=dm.Radau(num_segments=num_segments,
                                                order=transcription_order,
                                                segment_ends=seg_ends))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0.0, 100.0), duration_bounds=(0., 100.))

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s')

        phase.add_input_parameter('m', val=[[1, 2], [3, 4]], units='kg')

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interpolate(ys=[20, 0], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, -5], nodes='state_input')

        p.run_model()

        expected = np.array([[1, 2], [3, 4]])
        assert_rel_error(self, p.get_val('phase0.rhs_all.sum.m'), expected)

    def test_dynamic_input_parameter_connections_gl(self):

        @dm.declare_time(units='s')
        @dm.declare_state('v', rate_source='eom.v_dot', units='m/s')
        @dm.declare_state('h', rate_source='eom.h_dot', units='m')
        @dm.declare_parameter('m', targets='sum.m', units='kg', shape=(2, 2))
        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'value': np.zeros((nn, 2, 2))},
                                                      m_tot={'value': np.zeros(nn)}))

                self.add_subsystem('eom', FlightPathEOM2D(num_nodes=nn))

                self.connect('sum.m_tot', 'eom.m')

        optimizer = 'SLSQP'
        num_segments = 1
        transcription_order = 5

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring()

        seg_ends, _ = lgl(num_segments + 1)

        phase = dm.Phase(ode_class=TrajectoryODE,
                         transcription=dm.GaussLobatto(num_segments=num_segments,
                                                       order=transcription_order,
                                                       segment_ends=seg_ends))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0.0, 100.0), duration_bounds=(0., 100.))

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s')

        phase.add_input_parameter('m', val=[[1, 2], [3, 4]], units='kg')

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interpolate(ys=[20, 0], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, -5], nodes='state_input')

        p.run_model()

        gd = p.model.phase0.options['transcription'].grid_data

        expected = np.broadcast_to(np.array([[1, 2], [3, 4]]),
                                   (gd.subset_num_nodes['state_disc'], 2, 2))
        assert_rel_error(self, p.get_val('phase0.rhs_disc.sum.m'), expected)

        expected = np.broadcast_to(np.array([[1, 2], [3, 4]]),
                                   (gd.subset_num_nodes['col'], 2, 2))
        assert_rel_error(self, p.get_val('phase0.rhs_col.sum.m'), expected)

    def test_static_input_parameter_connections_gl(self):

        @dm.declare_time(units='s')
        @dm.declare_state('v', rate_source='eom.v_dot', units='m/s')
        @dm.declare_state('h', rate_source='eom.h_dot', units='m')
        @dm.declare_parameter('m', targets='sum.m', units='kg', shape=(2, 2), dynamic=False)
        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'value': np.zeros((2, 2))},
                                                      m_tot={'value': np.zeros(nn)}))

                self.add_subsystem('eom', FlightPathEOM2D(num_nodes=nn))

                self.connect('sum.m_tot', 'eom.m')

        optimizer = 'SLSQP'
        num_segments = 1
        transcription_order = 5

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring()

        seg_ends, _ = lgl(num_segments + 1)

        phase = dm.Phase(ode_class=TrajectoryODE,
                         transcription=dm.GaussLobatto(num_segments=num_segments,
                                                       order=transcription_order,
                                                       segment_ends=seg_ends))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0.0, 100.0), duration_bounds=(0., 100.))

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s')

        phase.add_input_parameter('m', val=[[1, 2], [3, 4]], units='kg')

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interpolate(ys=[20, 0], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, -5], nodes='state_input')

        p.run_model()

        expected = np.array([[1, 2], [3, 4]])
        assert_rel_error(self, p.get_val('phase0.rhs_disc.sum.m'), expected)
        assert_rel_error(self, p.get_val('phase0.rhs_col.sum.m'), expected)

if __name__ == '__main__':
    unittest.main()
