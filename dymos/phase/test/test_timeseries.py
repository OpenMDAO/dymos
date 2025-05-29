import unittest
import warnings

import numpy as np

from scipy.interpolate import interp1d

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

from dymos.models.atmosphere import USatm1976Comp
from dymos.examples.min_time_climb.aero import AeroGroup
from dymos.examples.min_time_climb.prop import PropGroup
from dymos.models.eom import FlightPathEOM2D


@use_tempdirs
class TestTimeseriesOutput(unittest.TestCase):

    def test_timeseries_gl(self, test_smaller_timeseries=False):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        if test_smaller_timeseries:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)
        else:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665, include_timeseries=True)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        phase.set_time_val(initial=0, duration=2.0)
        phase.set_state_val('x', (0, 10))
        phase.set_state_val('y', (10, 5))
        phase.set_state_val('v', (0, 9.9))
        phase.set_control_val('theta', (5, 100))
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        gd = phase.options['transcription'].grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']
        col_idxs = gd.subset_node_indices['col']

        assert_near_equal(p.get_val('phase0.t'),
                          p.get_val('phase0.timeseries.time')[:, 0])

        assert_near_equal(p.get_val('phase0.t_phase'),
                          p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_near_equal(p.get_val(f'phase0.states:{state}'),
                              p.get_val(f'phase0.timeseries.{state}')[state_input_idxs])

            assert_near_equal(p.get_val(f'phase0.state_interp.state_col:{state}'),
                              p.get_val(f'phase0.timeseries.{state}')[col_idxs])

        for control in ('theta',):
            assert_near_equal(p.get_val(f'phase0.controls:{control}'),
                              p.get_val(f'phase0.timeseries.{control}')[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                if test_smaller_timeseries:
                    with self.assertRaises(KeyError):
                        p.get_val(f'phase0.timeseries.{dp}')
                else:
                    assert_near_equal(p.get_val(f'phase0.parameter_vals:{dp}')[0],
                                      p.get_val(f'phase0.timeseries.{dp}')[i])

        # test simulation
        exp_out = phase.simulate()
        if test_smaller_timeseries:
            with self.assertRaises(KeyError):
                exp_out.get_val(f'phase0.timeseries.{dp}')
        else:  # no error accessing timseries.parameter
            exp_out.get_val(f'phase0.timeseries.{dp}')

    def test_timeseries_gl_smaller_timeseries(self):
        self.test_timeseries_gl(test_smaller_timeseries=True)

    def test_timeseries_radau(self, test_smaller_timeseries=False):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=8, order=3, compressed=True))

        phase.timeseries_options['include_state_rates'] = True

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        if test_smaller_timeseries:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)
        else:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665, include_timeseries=True)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        phase.timeseries_options['include_state_rates'] = True

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        phase.set_time_val(initial=0, duration=2.0)
        phase.set_state_val('x', (0, 10))
        phase.set_state_val('y', (10, 5))
        phase.set_state_val('v', (0, 9.9))
        phase.set_control_val('theta', (5, 100))
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        gd = phase.options['transcription'].grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']

        assert_near_equal(p.get_val('phase0.t'),
                          p.get_val('phase0.timeseries.time')[:, 0])

        assert_near_equal(p.get_val('phase0.t_phase'),
                          p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_near_equal(p.get_val(f'phase0.states:{state}'),
                              p.get_val(f'phase0.timeseries.{state}')[state_input_idxs])

        for control in ('theta',):
            assert_near_equal(p.get_val(f'phase0.controls:{control}'),
                              p.get_val(f'phase0.timeseries.{control}')[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                if test_smaller_timeseries:
                    with self.assertRaises(KeyError):
                        p.get_val(f'phase0.timeseries.{dp}')
                else:
                    assert_near_equal(p.get_val(f'phase0.parameters:{dp}')[0],
                                      p.get_val(f'phase0.timeseries.{dp}')[i])

        # test simulation
        exp_out = phase.simulate()
        if test_smaller_timeseries:
            with self.assertRaises(KeyError):
                exp_out.get_val(f'phase0.timeseries.{dp}')
        else:  # no error accessing timseries.parameter
            exp_out.get_val(f'phase0.timeseries.{dp}')

        # Test that the state rates are output in both the radau and solveivp timeseries outputs
        t_sol = p.get_val('phase0.timeseries.time')
        t_sim = exp_out.get_val('phase0.timeseries.time')

        p.model.list_outputs()

        for state_name, rate_name in (('x', 'xdot'), ('y', 'ydot'), ('v', 'vdot')):
            rate_sol = p.get_val(f'phase0.timeseries.{rate_name}')
            rate_sim = exp_out.get_val(f'phase0.timeseries.{rate_name}')
            rate_t_sim = interp1d(t_sim.ravel(), rate_sim.ravel())
            assert_near_equal(rate_t_sim(t_sol), rate_sol, tolerance=1.0E-3)

    def test_timeseries_radau_smaller_timeseries(self):
        self.test_timeseries_radau(test_smaller_timeseries=True)

    def test_timeseries_explicit_shooting(self, test_smaller_timeseries=False):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        tx = dm.ExplicitShooting(grid=dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=5))

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

        # automatically discover states
        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
        phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9,
                          ref=90., rate2_continuity=True)

        phase.add_boundary_constraint('x', loc='final', equals=10.0)
        phase.add_boundary_constraint('y', loc='final', equals=5.0)

        p.model.add_subsystem('phase0', phase)

        phase.add_objective('time', loc='final')

        p.setup(force_alloc_complex=True)

        phase.set_time_val(initial=0, duration=2.0)
        phase.set_state_val('x', (0, 10))
        phase.set_state_val('y', (10, 5))
        phase.set_state_val('v', (0, 9.9))
        phase.set_control_val('theta', (5, 100))
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        assert_near_equal(np.atleast_2d(p.get_val('phase0.t')).T,
                          p.get_val('phase0.timeseries.time'))

        for state in ('x', 'y', 'v'):
            assert_near_equal(p.get_val(f'phase0.integrator.states_out:{state}'),
                              p.get_val(f'phase0.timeseries.{state}'))

        for control in ('theta',):
            assert_near_equal(p.get_val(f'phase0.control_values:{control}'),
                              p.get_val(f'phase0.timeseries.{control}'))


class MinTimeClimbODEDuplicateOutput(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['h'])

        self.add_subsystem(name='aero',
                           subsys=AeroGroup(num_nodes=nn),
                           promotes_inputs=['v', 'alpha', 'S'])

        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop',
                           subsys=PropGroup(num_nodes=nn),
                           promotes_inputs=['h', 'Isp', 'throttle'])

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightPathEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'gam', 'alpha'])

        foo = self.add_subsystem('foo', om.IndepVarComp())
        foo.add_output('rho', val=100 * np.ones(nn), units='g/cm**3')

        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('prop.thrust', 'flight_dynamics.T')


@require_pyoptsparse(optimizer='SLSQP')
def min_time_climb(num_seg=3, transcription_class=dm.Radau, transcription_order=3,
                   force_alloc_complex=False, timeseries_expr=False):

    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()

    tx = transcription_class(num_segments=num_seg, order=transcription_order)

    traj = dm.Trajectory()

    ode_class = MinTimeClimbODEDuplicateOutput if not timeseries_expr else MinTimeClimbODE

    phase = dm.Phase(ode_class=ode_class, transcription=tx)
    traj.add_phase('phase0', phase)

    p.model.add_subsystem('traj', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                           duration_ref=100.0)

    phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6,
                    ref=1.0E3, defect_ref=1.0E3, units='m',
                    rate_source='flight_dynamics.r_dot')

    phase.add_state('h', fix_initial=True, lower=0, upper=20000.0,
                    ref=1.0E2, defect_ref=1.0E2, units='m',
                    rate_source='flight_dynamics.h_dot', targets=['h'])

    phase.add_state('v', fix_initial=True, lower=10.0,
                    ref=1.0E2, defect_ref=1.0E2, units='m/s',
                    rate_source='flight_dynamics.v_dot', targets=['v'])

    phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5,
                    ref=1.0, defect_ref=1.0, units='rad',
                    rate_source='flight_dynamics.gam_dot', targets=['gam'])

    phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5,
                    ref=1.0E3, defect_ref=1.0E3, units='kg',
                    rate_source='prop.m_dot', targets=['m'])

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      rate_continuity=True, rate_continuity_scaler=100.0,
                      rate2_continuity=False, targets=['alpha'])

    phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
    phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
    phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3)
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0)

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Unnecessary but included to test capability
    phase.add_path_constraint(name='alpha', lower=-8, upper=8)
    phase.add_path_constraint(name='time', lower=0, upper=400)
    phase.add_path_constraint(name='time_phase', lower=0, upper=400)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=1.0)

    # Add all ODE outputs to the timeseries
    phase.add_timeseries_output('*')
    if timeseries_expr:
        phase.add_timeseries_output('R = atmos.pres / (atmos.temp * atmos.rho)')

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    phase.set_time_val(initial=0, duration=300)
    phase.set_state_val('r', [0.0, 111319.54])
    phase.set_state_val('h', [100.0, 20000.0])
    phase.set_state_val('v', [135.964, 283.159])
    phase.set_state_val('gam', [0.0, 0.0])
    phase.set_state_val('m', [19030.468, 16841.431])
    phase.set_control_val('alpha', [0.0, 0.0])

    return p


@use_tempdirs
class TestDuplicateTimeseriesGlobName(unittest.TestCase):

    def test_duplicate_timeseries_glob_name_radau(self):
        """
        Test that the user gets a warning about multiple timeseries with the same name.
        """
        msg = "Error during configure_timeseries_output_introspection in phase traj.phases.phase0."
        with self.assertRaises(RuntimeError) as e:
            min_time_climb(num_seg=12, transcription_class=dm.Radau, transcription_order=3)
        self.assertEqual(str(e.exception), msg)

    def test_duplicate_timeseries_glob_name_gl(self):
        """
        Test that the user gets a warning about multiple timeseries with the same name.
        """
        msg = "Error during configure_timeseries_output_introspection in phase traj.phases.phase0."
        with self.assertRaises(RuntimeError) as e:
            min_time_climb(num_seg=12, transcription_class=dm.GaussLobatto, transcription_order=3)
        self.assertEqual(str(e.exception), msg)


@use_tempdirs
class TestTimeseriesExprBrachistochrone(unittest.TestCase):

    @staticmethod
    def make_problem_brachistochrone(transcription, polynomial_control=False):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=transcription)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        if polynomial_control:
            phase.add_control('theta', order=1, units='deg', lower=0.01, upper=179.9, control_type='polynomial')
            control_name = 'controls:theta'
        else:
            phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                              units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)
            control_name = 'controls:theta'
        phase.add_boundary_constraint('x', loc='final', equals=10.0)
        phase.add_boundary_constraint('y', loc='final', equals=5.0)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665, include_timeseries=True)

        phase.add_objective('time_phase', loc='final', scaler=10)
        phase.add_timeseries_output('z=x*y + x**2', units='m**2')
        phase.add_timeseries_output('f=3*g*cos(theta)**2', units='deg**2')

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = om.DirectSolver()
        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)
        return p

    def test_invalid_expr_var(self):
        p = om.Problem(model=om.Group())

        tx = dm.Radau(num_segments=5, order=3, compressed=True)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665, include_timeseries=True)

        phase.add_timeseries_output('k=units', units='m**2')

        with self.assertRaises(NameError) as e:
            p.setup(check=True)

        expected = ("'phase0.rhs_all.exec_comp' <class ExecComp>: cannot use variable "
                    "name 'units' because it's a reserved keyword.")
        self.assertEqual(str(e.exception), expected)

    def test_input_units(self):
        p = om.Problem(model=om.Group())

        tx = dm.Radau(num_segments=5, order=3, compressed=True)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665, include_timeseries=True)

        phase.add_timeseries_output('sin_theta=sin(theta)', units='unitless', theta={'units': 'rad'})

        p.setup(check=True)

        phase.set_time_val(initial=0, duration=2.0)
        phase.set_state_val('x', (0, 10))
        phase.set_state_val('y', (10, 5))
        phase.set_state_val('v', (0, 9.9))
        phase.set_control_val('theta', (5, 100))
        phase.set_parameter_val('g', 9.80665)

        p.run_model()

        sin_theta = p.get_val('phase0.timeseries.sin_theta')
        theta = p.get_val('phase0.timeseries.theta', units='rad')

        assert_near_equal(np.sin(theta), sin_theta)

    def test_output_units(self):
        p = om.Problem(model=om.Group())

        tx = dm.Radau(num_segments=5, order=3, compressed=True)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665, include_timeseries=True)

        phase.add_timeseries_output('sin_theta=sin(theta)', shape=(1,), units='unitless', theta={'units': 'rad', 'shape': (1,)})
        phase.add_timeseries_output('sin_theta2=sin(theta)', sin_theta2={'units': 'unitless'})
        phase.add_timeseries_output('sin_theta3=sin(theta)', shape=(1,), sin_theta3={'shape': (1,)})

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)
        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.run_model()

        var_data = p.model.list_vars(units=True, shape=True, out_stream=None)
        var_data = {meta['prom_name']: meta for abs_path, meta in var_data}

        self.assertEqual('unitless', var_data['phase0.timeseries.sin_theta']['units'])
        self.assertEqual('unitless', var_data['phase0.timeseries.sin_theta']['units'])

    def test_timeseries_expr_radau(self):
        tx = dm.Radau(num_segments=5, order=3, compressed=True)
        p = self.make_problem_brachistochrone(transcription=tx)
        p.run_driver()

        x = p.get_val('phase0.timeseries.x')
        y = p.get_val('phase0.timeseries.y')
        theta = p.get_val('phase0.timeseries.theta')
        g = p.get_val('phase0.timeseries.g')

        z_computed = x * y + x**2
        f_computed = 3 * g * np.cos(theta)**2
        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')
        assert_near_equal(z_computed, z_ts, tolerance=1e-12)
        assert_near_equal(f_computed, f_ts, tolerance=1e-12)

    def test_timeseries_expr_radau_polynomial_control(self):
        tx = dm.Radau(num_segments=5, order=3, compressed=True)
        p = self.make_problem_brachistochrone(transcription=tx, polynomial_control=True)
        p.run_driver()

        x = p.get_val('phase0.timeseries.x')
        y = p.get_val('phase0.timeseries.y')
        theta = p.get_val('phase0.timeseries.theta')
        g = p.get_val('phase0.timeseries.g')

        z_computed = x * y + x**2
        f_computed = 3 * g * np.cos(theta)**2
        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')
        assert_near_equal(z_computed, z_ts, tolerance=1e-12)
        assert_near_equal(f_computed, f_ts, tolerance=1e-12)

    def test_timeseries_expr_gl(self):
        tx = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        p = self.make_problem_brachistochrone(transcription=tx)
        p.run_driver()
        x = p.get_val('phase0.timeseries.x')
        y = p.get_val('phase0.timeseries.y')
        theta = p.get_val('phase0.timeseries.theta')
        g = p.get_val('phase0.timeseries.g')

        z_computed = x * y + x**2
        f_computed = 3 * g * np.cos(theta) ** 2

        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')
        assert_near_equal(z_computed, z_ts, tolerance=1e-12)
        assert_near_equal(f_computed, f_ts, tolerance=1e-12)

    def test_timeseries_expr_birkhoff(self):
        tx = dm.Birkhoff(num_nodes=21)
        p = self.make_problem_brachistochrone(transcription=tx)
        p.run_driver()
        x = p.get_val('phase0.timeseries.x')
        y = p.get_val('phase0.timeseries.y')
        theta = p.get_val('phase0.timeseries.theta')
        g = p.get_val('phase0.timeseries.g')

        z_computed = x * y + x**2
        f_computed = 3 * g * np.cos(theta) ** 2

        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')
        assert_near_equal(z_computed, z_ts, tolerance=1e-12)
        assert_near_equal(f_computed, f_ts, tolerance=1e-12)

    def test_timeseries_expr_gl_polynomial_control(self):
        tx = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        p = self.make_problem_brachistochrone(transcription=tx, polynomial_control=True)
        p.run_driver()

        x = p.get_val('phase0.timeseries.x')
        y = p.get_val('phase0.timeseries.y')
        theta = p.get_val('phase0.timeseries.theta')
        g = p.get_val('phase0.timeseries.g')

        z_computed = x * y + x**2
        f_computed = 3 * g * np.cos(theta)**2
        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')
        assert_near_equal(z_computed, z_ts, tolerance=1e-12)
        assert_near_equal(f_computed, f_ts, tolerance=1e-12)

    def test_timeseries_expr_explicit_shooting(self):

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=om.UnitsWarning)
            tx = dm.ExplicitShooting(grid=dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=5, compressed=True))

            p = self.make_problem_brachistochrone(transcription=tx)
            p.run_driver()
            x = p.get_val('phase0.timeseries.x')
            y = p.get_val('phase0.timeseries.y')
            theta = p.get_val('phase0.timeseries.theta')
            g = p.get_val('phase0.timeseries.g')

            z_computed = x * y + x**2
            f_computed = 3 * g * np.cos(theta) ** 2

            z_ts = p.get_val('phase0.timeseries.z')
            f_ts = p.get_val('phase0.timeseries.f')
            assert_near_equal(z_computed, z_ts, tolerance=1e-12)
            assert_near_equal(f_computed, f_ts, tolerance=1e-12)

    def test_timeseries_expr_explicit_shooting_polynomial_controls(self):
        tx = dm.ExplicitShooting(grid=dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=5, compressed=True))

        p = self.make_problem_brachistochrone(transcription=tx, polynomial_control=True)
        p.run_driver()
        x = p.get_val('phase0.timeseries.x')
        y = p.get_val('phase0.timeseries.y')
        theta = p.get_val('phase0.timeseries.theta')
        g = p.get_val('phase0.timeseries.g')

        z_computed = x * y + x**2
        f_computed = 3 * g * np.cos(theta) ** 2

        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')
        assert_near_equal(z_computed, z_ts, tolerance=1e-12)
        assert_near_equal(f_computed, f_ts, tolerance=1e-12)

    def test_timeseries_expr_solve_ivp(self):
        tx = dm.Radau(num_segments=5, order=3, compressed=True)

        p = self.make_problem_brachistochrone(transcription=tx)
        p.run_driver()

        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')

        phase0 = p.model._get_subsystem('phase0')
        sim = phase0.simulate(times_per_seg=30)

        z_sim = sim.get_val('phase0.timeseries.z')
        f_sim = sim.get_val('phase0.timeseries.f')
        assert_near_equal(z_ts[-1], z_sim[-1], tolerance=1e-3)
        assert_near_equal(f_ts[-1], f_sim[-1], tolerance=1e-3)

    def test_timeseries_expr_solve_ivp_polynomial_controls(self):
        tx = dm.Radau(num_segments=5, order=3, compressed=True)

        p = self.make_problem_brachistochrone(transcription=tx, polynomial_control=True)
        p.run_driver()

        z_ts = p.get_val('phase0.timeseries.z')
        f_ts = p.get_val('phase0.timeseries.f')

        phase0 = p.model._get_subsystem('phase0')
        sim = phase0.simulate(times_per_seg=30)

        z_sim = sim.get_val('phase0.timeseries.z')
        f_sim = sim.get_val('phase0.timeseries.f')
        assert_near_equal(z_ts[-1], z_sim[-1], tolerance=1e-3)
        assert_near_equal(f_ts[-1], f_sim[-1], tolerance=1e-3)


@use_tempdirs
class TestTimeseriesExprMinTimeClimb(unittest.TestCase):

    def test_timeseries_expr_radau(self):
        p = min_time_climb(transcription_class=dm.Radau, num_seg=12, transcription_order=3, timeseries_expr=True)
        p.run_model()

        pres = p.get_val('traj.phase0.timeseries.pres')
        temp = p.get_val('traj.phase0.timeseries.temp')
        rho = p.get_val('traj.phase0.timeseries.rho')

        R_computed = pres / (temp * rho)
        R_ts = p.get_val('traj.phase0.timeseries.R')
        assert_near_equal(R_computed, R_ts, tolerance=1e-12)

    def test_timeseries_expr_gl(self):
        p = min_time_climb(transcription_class=dm.GaussLobatto, num_seg=12,
                           transcription_order=3, timeseries_expr=True)
        p.run_driver()

        pres = p.get_val('traj.phase0.timeseries.pres')
        temp = p.get_val('traj.phase0.timeseries.temp')
        rho = p.get_val('traj.phase0.timeseries.rho')

        R_computed = pres / (temp * rho)
        R_ts = p.get_val('traj.phase0.timeseries.R')
        assert_near_equal(R_computed, R_ts, tolerance=1e-12)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
