import unittest

import numpy as np

from scipy.interpolate import interp1d

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

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
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665, include_timeseries=False)
        else:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        gd = phase.options['transcription'].grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']
        col_idxs = gd.subset_node_indices['col']

        assert_near_equal(p.get_val('phase0.time'),
                          p.get_val('phase0.timeseries.time')[:, 0])

        assert_near_equal(p.get_val('phase0.time_phase'),
                          p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_near_equal(p.get_val('phase0.states:{0}'.format(state)),
                              p.get_val('phase0.timeseries.states:'
                                        '{0}'.format(state))[state_input_idxs])

            assert_near_equal(p.get_val('phase0.state_interp.state_col:{0}'.format(state)),
                              p.get_val('phase0.timeseries.states:'
                                        '{0}'.format(state))[col_idxs])

        for control in ('theta',):
            assert_near_equal(p.get_val('phase0.controls:{0}'.format(control)),
                              p.get_val('phase0.timeseries.controls:'
                                        '{0}'.format(control))[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                if test_smaller_timeseries:
                    with self.assertRaises(KeyError):
                        p.get_val('phase0.timeseries.parameters:{0}'.format(dp))
                else:
                    assert_near_equal(p.get_val('phase0.parameters:{0}'.format(dp))[0],
                                      p.get_val('phase0.timeseries.parameters:{0}'.format(dp))[i])

        # call simulate to test SolveIVP transcription
        exp_out = phase.simulate()
        if test_smaller_timeseries:
            with self.assertRaises(KeyError):
                exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))
        else:  # no error accessing timseries.parameter
            exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))

    def test_timeseries_gl_smaller_timeseries(self):
        self.test_timeseries_gl(test_smaller_timeseries=True)

    def test_timeseries_radau(self, test_smaller_timeseries=False):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        if test_smaller_timeseries:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665, include_timeseries=False)
        else:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interp('x', [0, 10])
        p['phase0.states:y'] = phase.interp('y', [10, 5])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        gd = phase.options['transcription'].grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']

        assert_near_equal(p.get_val('phase0.time'),
                          p.get_val('phase0.timeseries.time')[:, 0])

        assert_near_equal(p.get_val('phase0.time_phase'),
                          p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_near_equal(p.get_val('phase0.states:{0}'.format(state)),
                              p.get_val('phase0.timeseries.states:'
                                        '{0}'.format(state))[state_input_idxs])

        for control in ('theta',):
            assert_near_equal(p.get_val('phase0.controls:{0}'.format(control)),
                              p.get_val('phase0.timeseries.controls:'
                                        '{0}'.format(control))[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                if test_smaller_timeseries:
                    with self.assertRaises(KeyError):
                        p.get_val('phase0.timeseries.parameters:{0}'.format(dp))
                else:
                    assert_near_equal(p.get_val('phase0.parameters:{0}'.format(dp))[0],
                                      p.get_val('phase0.timeseries.parameters:'
                                                '{0}'.format(dp))[i])

        # call simulate to test SolveIVP transcription
        exp_out = phase.simulate()
        if test_smaller_timeseries:
            with self.assertRaises(KeyError):
                exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))
        else:  # no error accessing timseries.parameter
            exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))

        # Test that the state rates are output in both the radau and solveivp timeseries outputs
        t_sol = p.get_val('phase0.timeseries.time')
        t_sim = exp_out.get_val('phase0.timeseries.time')

        for state_name in ('x', 'y', 'v'):
            rate_sol = p.get_val(f'phase0.timeseries.state_rates:{state_name}')
            rate_sim = exp_out.get_val(f'phase0.timeseries.state_rates:{state_name}')

            rate_t_sim = interp1d(t_sim.ravel(), rate_sim.ravel())

            assert_near_equal(rate_t_sim(t_sol), rate_sol, tolerance=1.0E-3)

    def test_timeseries_radau_smaller_timeseries(self):
        self.test_timeseries_radau(test_smaller_timeseries=True)


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


def min_time_climb(num_seg=3, transcription_class=dm.Radau, transcription_order=3,
                   force_alloc_complex=False):

    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()

    tx = transcription_class(num_segments=num_seg, order=transcription_order)

    traj = dm.Trajectory()

    phase = dm.Phase(ode_class=MinTimeClimbODEDuplicateOutput, transcription=tx)
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

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 300.0

    p['traj.phase0.states:r'] = phase.interp('r', [0.0, 111319.54])
    p['traj.phase0.states:h'] = phase.interp('h', [100.0, 20000.0])
    p['traj.phase0.states:v'] = phase.interp('v', [135.964, 283.159])
    p['traj.phase0.states:gam'] = phase.interp('gam', [0.0, 0.0])
    p['traj.phase0.states:m'] = phase.interp('m', [19030.468, 16841.431])
    p['traj.phase0.controls:alpha'] = phase.interp('alpha', [0.0, 0.0])

    return p


class TestDuplicateTimeseriesGlobName(unittest.TestCase):

    def test_duplicate_timeseries_glob_name(self):
        """
        Test that the user gets a warning about multiple timeseries with the same name.
        """

        msg = "The timeseries variable name rho is duplicated in these variables: atmos.rho, " \
              "foo.rho. Disambiguate by using the add_timeseries_output output_name option."
        with assert_warning(UserWarning, msg):
            p = min_time_climb(num_seg=12, transcription_class=dm.Radau, transcription_order=3)

        with assert_warning(UserWarning, msg):
            p = min_time_climb(num_seg=12, transcription_class=dm.GaussLobatto,
                               transcription_order=3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
