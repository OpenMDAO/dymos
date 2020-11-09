import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from openmdao.utils.testing_utils import use_tempdirs


def min_time_climb(optimizer='SLSQP', num_seg=3, transcription='gauss-lobatto',
                   transcription_order=3, force_alloc_complex=False):

    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()

    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Function precision'] = 1.0E-12
        p.driver.opt_settings['Linesearch tolerance'] = 0.1
        p.driver.opt_settings['Major step limit'] = 0.5

    t = {'gauss-lobatto': dm.GaussLobatto(num_segments=num_seg, order=transcription_order),
         'radau-ps': dm.Radau(num_segments=num_seg, order=transcription_order),
         'runge-kutta': dm.RungeKutta(num_segments=num_seg)}

    traj = dm.Trajectory()

    phase = dm.Phase(ode_class=MinTimeClimbODE, transcription=t[transcription])
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

    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Unnecessary but included to test capability
    phase.add_path_constraint(name='alpha', units='deg', lower=-8, upper=8)
    phase.add_path_constraint(name='time', lower=0, upper=400)
    phase.add_path_constraint(name='time_phase', lower=0, upper=400)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=1.0)

    # test mixing wildcard ODE variable expansion and unit overrides
    phase.add_timeseries_output(['aero.*', 'prop.thrust', 'prop.m_dot'],
                                units={'aero.f_lift': 'lbf', 'prop.thrust': 'lbf'})

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 300.0

    p['traj.phase0.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='state_input')
    p['traj.phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
    p['traj.phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
    p['traj.phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
    p['traj.phase0.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='state_input')
    p['traj.phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

    p.run_driver()

    return p


@use_tempdirs
class TestMinTimeClimb(unittest.TestCase):

    def test_results_gauss_lobatto(self):
        p = min_time_climb(optimizer='SLSQP', num_seg=12, transcription_order=3,
                           transcription='gauss-lobatto')

        # Check that time matches to within 1% of an externally verified solution.
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 321.0, tolerance=0.02)

        # Verify that ODE output mach is added to the timeseries
        assert_near_equal(p.get_val('traj.phase0.timeseries.mach')[-1], 1.0, tolerance=1.0E-2)

        # verify all wildcard timeseries exist
        output_dict = dict(p.model.list_outputs(units=True))
        ts = [k for k, v in output_dict.items() if 'timeseries' in k]
        for c in ['mach', 'CD0', 'kappa', 'CLa', 'CL', 'CD', 'q', 'f_lift', 'f_drag', 'thrust', 'm_dot']:
            assert(any([True for t in ts if 'timeseries.' + c in t]))
        # verify time series units
        assert(output_dict['traj.phases.phase0.timeseries.thrust']['units'] == 'lbf')  # no wildcard, from units dict
        assert(output_dict['traj.phases.phase0.timeseries.m_dot']['units'] == 'kg/s')  # no wildcard, from ODE
        assert(output_dict['traj.phases.phase0.timeseries.f_drag']['units'] == 'N')    # wildcard, from ODE
        assert(output_dict['traj.phases.phase0.timeseries.f_lift']['units'] == 'lbf')  # wildcard, from units dict

    def test_results_radau(self):
        p = min_time_climb(optimizer='SLSQP', num_seg=12, transcription_order=3,
                           transcription='radau-ps')

        # Check that time matches to within 1% of an externally verified solution.
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 321.0, tolerance=0.02)

        # Verify that ODE output mach is added to the timeseries
        assert_near_equal(p.get_val('traj.phase0.timeseries.mach')[-1], 1.0, tolerance=1.0E-2)

        # verify all wildcard timeseries exist
        output_dict = dict(p.model.list_outputs(units=True))
        ts = [k for k, v in output_dict.items() if 'timeseries' in k]
        for c in ['mach', 'CD0', 'kappa', 'CLa', 'CL', 'CD', 'q', 'f_lift', 'f_drag', 'thrust', 'm_dot']:
            assert(any([True for t in ts if 'timeseries.' + c in t]))
        # verify time series units
        assert(output_dict['traj.phases.phase0.timeseries.thrust']['units'] == 'lbf')  # no wildcard, from units dict
        assert(output_dict['traj.phases.phase0.timeseries.m_dot']['units'] == 'kg/s')  # no wildcard, from ODE
        assert(output_dict['traj.phases.phase0.timeseries.f_drag']['units'] == 'N')    # wildcard, from ODE
        assert(output_dict['traj.phases.phase0.timeseries.f_lift']['units'] == 'lbf')  # wildcard, from units dict


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
