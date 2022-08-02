import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.utils.lgl import lgl
from dymos.models.eom import FlightPathEOM2D


@use_tempdirs
class TestParameterConnections(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_dynamic_parameter_connections_radau(self):

        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'val': np.zeros((nn, 2, 2)),
                                                         'units': 'kg'},
                                                      m_tot={'val': np.zeros(nn),
                                                             'units': 'kg'}))

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

        phase.set_time_options(initial_bounds=(0.0, 100.0), duration_bounds=(0., 100.), units='s')

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m', rate_source='eom.h_dot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='eom.v_dot')

        phase.add_parameter('m', val=[[1, 2], [3, 4]], units='kg', targets='sum.m')

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interp('h', [20, 0])
        p['phase0.states:v'] = phase.interp('v', [0, -5])

        p.run_model()

        expected = np.broadcast_to(np.array([[1, 2], [3, 4]]),
                                   (p.model.phase0.options['transcription'].grid_data.num_nodes, 2, 2))
        assert_near_equal(p.get_val('phase0.rhs_all.sum.m'), expected)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_static_parameter_connections_radau(self):

        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'val': np.zeros((2, 2)),
                                                         'units': 'kg'},
                                                      m_tot={'val': np.zeros(nn),
                                                             'units': 'kg'}))

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

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m', rate_source='eom.h_dot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='eom.v_dot')

        phase.add_parameter('m', val=[[1, 2], [3, 4]], units='kg', targets='sum.m', static_target=True)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interp('h', [20, 0])
        p['phase0.states:v'] = phase.interp('v', [0, -5])

        p.run_model()

        expected = np.array([[1, 2], [3, 4]])
        assert_near_equal(p.get_val('phase0.rhs_all.sum.m'), expected)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_dynamic_parameter_connections_gl(self):

        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'val': np.zeros((nn, 2, 2)),
                                                         'units': 'kg'},
                                                      m_tot={'val': np.zeros(nn),
                                                             'units': 'kg'}))

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

        phase.set_time_options(initial_bounds=(0.0, 100.0), duration_bounds=(0., 100.), units='s')

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m', rate_source='eom.h_dot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='eom.v_dot')

        phase.add_parameter('m', val=[[1, 2], [3, 4]], units='kg', targets='sum.m')

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interp('h', [20, 0])
        p['phase0.states:v'] = phase.interp('v', [0, -5])

        p.run_model()

        gd = p.model.phase0.options['transcription'].grid_data

        expected = np.broadcast_to(np.array([[1, 2], [3, 4]]),
                                   (gd.subset_num_nodes['state_disc'], 2, 2))
        assert_near_equal(p.get_val('phase0.rhs_disc.sum.m'), expected)

        expected = np.broadcast_to(np.array([[1, 2], [3, 4]]),
                                   (gd.subset_num_nodes['col'], 2, 2))
        assert_near_equal(p.get_val('phase0.rhs_col.sum.m'), expected)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_static_parameter_connections_gl(self):

        class TrajectoryODE(om.Group):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('sum', om.ExecComp('m_tot = sum(m)',
                                                      m={'val': np.zeros((2, 2)),
                                                         'units': 'kg'},
                                                      m_tot={'val': np.zeros(nn),
                                                             'units': 'kg'}))

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

        phase.set_time_options(initial_bounds=(0.0, 100.0), duration_bounds=(0., 100.), units='s')

        phase.add_state('h', fix_initial=True, fix_final=True, lower=0.0, units='m', rate_source='eom.h_dot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='eom.v_dot')

        phase.add_parameter('m', val=[[1, 2], [3, 4]], units='kg', targets='sum.m', static_target=True)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0

        p['phase0.states:h'] = phase.interp('h', [20, 0])
        p['phase0.states:v'] = phase.interp('v', [0, -5])

        p.run_model()

        expected = np.array([[1, 2], [3, 4]])
        assert_near_equal(p.get_val('phase0.rhs_disc.sum.m'), expected)
        assert_near_equal(p.get_val('phase0.rhs_col.sum.m'), expected)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
