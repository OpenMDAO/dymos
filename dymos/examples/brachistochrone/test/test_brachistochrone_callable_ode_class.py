import os
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm


@use_tempdirs
class TestBrachExecCompODE(unittest.TestCase):

    def _make_problem(self, transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                      compressed=True, optimizer='SLSQP', run_driver=True, force_alloc_complex=False,
                      solve_segments=False):

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring(tol=1.0E-12)

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=num_segments,
                         order=transcription_order,
                         compressed=compressed)

        ode = lambda num_nodes: om.ExecComp(['vdot = g * cos(theta)',
                                             'xdot = v * sin(theta)',
                                             'ydot = -v * cos(theta)'],
                                            g={'value': 9.80665, 'units': 'm/s**2'},
                                            v={'shape': (num_nodes,), 'units': 'm/s'},
                                            theta={'shape': (num_nodes,), 'units': 'rad'},
                                            vdot={'shape': (num_nodes,),
                                                  'units': 'm/s**2',
                                                  'tags': ['state_rate_source:v']},
                                            xdot={'shape': (num_nodes,),
                                                  'units': 'm/s',
                                                  'tags': ['state_rate_source:x']},
                                            ydot={'shape': (num_nodes,),
                                                  'units': 'm/s',
                                                  'tags': ['state_rate_source:y']},
                                            has_diag_partials=True)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=ode, transcription=t)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=solve_segments)
        phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=solve_segments)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=solve_segments)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', dynamic=False)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['traj0.phase0.parameters:g'] = 9.80665

        dm.run_problem(p, run_driver=run_driver, simulate=True)

        return p

    def run_asserts(self):

        for db in ['dymos_solution.db', 'dymos_simulation.db']:
            p = om.CaseReader(db).get_case('final')

            t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
            tf = p.get_val('traj0.phase0.timeseries.time')[-1]

            x0 = p.get_val('traj0.phase0.timeseries.states:x')[0]
            xf = p.get_val('traj0.phase0.timeseries.states:x')[-1]

            y0 = p.get_val('traj0.phase0.timeseries.states:y')[0]
            yf = p.get_val('traj0.phase0.timeseries.states:y')[-1]

            v0 = p.get_val('traj0.phase0.timeseries.states:v')[0]
            vf = p.get_val('traj0.phase0.timeseries.states:v')[-1]

            g = p.get_val('traj0.phase0.timeseries.parameters:g')[0]

            thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1]

            assert_near_equal(t_initial, 0.0)
            assert_near_equal(x0, 0.0)
            assert_near_equal(y0, 10.0)
            assert_near_equal(v0, 0.0)

            assert_near_equal(tf, 1.8016, tolerance=0.01)
            assert_near_equal(xf, 10.0, tolerance=0.01)
            assert_near_equal(yf, 5.0, tolerance=0.01)
            assert_near_equal(vf, 9.902, tolerance=0.01)
            assert_near_equal(g, 9.80665, tolerance=0.01)

            assert_near_equal(thetaf, 100.12, tolerance=0.01)

    def test_ex_brachistochrone_radau_uncompressed(self):
        self._make_problem(transcription='radau-ps', compressed=False)
        self.run_asserts()

    def test_ex_brachistochrone_gl_uncompressed(self):
        self._make_problem(transcription='gauss-lobatto', compressed=False)
        self.run_asserts()


@use_tempdirs
class TestInvalidCallableODEClass(unittest.TestCase):

    def test_invalid_callable(self):
        num_segments = 10
        transcription_order = 3
        compressed = False

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring(tol=1.0E-12)

        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)

        ode = lambda num_nodes: num_nodes*2

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=ode, transcription=t)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False, rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=False, rate_source='ydot')

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False, rate_source='vdot')

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', dynamic=False)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        with self.assertRaises(expected_exception=ValueError) as e:
            p.setup()
        expected = "When provided as a callable, ode_class must return an instance of " \
                   "openmdao.core.System.  Got <class 'int'>"
        self.assertEqual(expected, str(e.exception))


class CallableBrachistochroneODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def __call__(self, num_nodes, **kwargs):
        from copy import deepcopy
        ret = deepcopy(self)
        ret.options['num_nodes'] = num_nodes
        return ret

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=9.80665, desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.ones(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s',
                        tags=['state_rate_source:x', 'state_units:m'])

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s',
                        tags=['state_rate_source:y', 'state_units:m'])

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2',
                        tags=['state_rate_source:v', 'state_units:m/s'])

        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta


@use_tempdirs
class TestBrachCallableODE(unittest.TestCase):

    def setUp(self):
        self.ode = CallableBrachistochroneODE(num_nodes=1)

    def _make_problem(self, transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                      compressed=True, optimizer='SLSQP', run_driver=True,
                      force_alloc_complex=False,
                      solve_segments=False):

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.declare_coloring(tol=1.0E-12)

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=num_segments,
                         order=transcription_order,
                         compressed=compressed)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=self.ode, transcription=t)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=solve_segments,
                        rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=solve_segments,
                        rate_source='ydot')

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=solve_segments,
                        rate_source='vdot')

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', dynamic=False)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

        p['traj0.phase0.t_initial'] = 0.0
        p['traj0.phase0.t_duration'] = 2.0

        p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['traj0.phase0.parameters:g'] = 9.80665

        dm.run_problem(p, run_driver=run_driver, simulate=True)

        return p

    def run_asserts(self):

        for db in ['dymos_solution.db', 'dymos_simulation.db']:
            p = om.CaseReader(db).get_case('final')

            t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
            tf = p.get_val('traj0.phase0.timeseries.time')[-1]

            x0 = p.get_val('traj0.phase0.timeseries.states:x')[0]
            xf = p.get_val('traj0.phase0.timeseries.states:x')[-1]

            y0 = p.get_val('traj0.phase0.timeseries.states:y')[0]
            yf = p.get_val('traj0.phase0.timeseries.states:y')[-1]

            v0 = p.get_val('traj0.phase0.timeseries.states:v')[0]
            vf = p.get_val('traj0.phase0.timeseries.states:v')[-1]

            g = p.get_val('traj0.phase0.timeseries.parameters:g')[0]

            thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1]

            assert_near_equal(t_initial, 0.0)
            assert_near_equal(x0, 0.0)
            assert_near_equal(y0, 10.0)
            assert_near_equal(v0, 0.0)

            assert_near_equal(tf, 1.8016, tolerance=0.01)
            assert_near_equal(xf, 10.0, tolerance=0.01)
            assert_near_equal(yf, 5.0, tolerance=0.01)
            assert_near_equal(vf, 9.902, tolerance=0.01)
            assert_near_equal(g, 9.80665, tolerance=0.01)

            assert_near_equal(thetaf, 100.12, tolerance=0.01)

    def test_ex_brachistochrone_radau_uncompressed(self):
        self._make_problem(transcription='radau-ps', compressed=False)
        self.run_asserts()

    def test_in_series(self):
        self._make_problem(transcription='gauss-lobatto', compressed=False)
        self._make_problem(transcription='radau-ps', compressed=False)
        self.run_asserts()
