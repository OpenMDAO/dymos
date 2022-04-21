import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
import dymos as dm
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


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
         'radau-ps': dm.Radau(num_segments=num_seg, order=transcription_order)}

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

    # test mixing wildcard ODE variable expansion and unit overrides
    phase.add_timeseries_output(['aero.*', 'prop.thrust', 'prop.m_dot'],
                                units={'aero.f_lift': 'lbf', 'prop.thrust': 'lbf'})

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

    dm.run_problem(p, simulate=True)

    return p


@use_tempdirs
class TestStateRef0(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def run_problem(self, tx):

        #
        # Instantiate the problem and configure the optimization driver
        #
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        # p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        #
        # Instantiate the trajectory and phase
        #
        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                        transcription=tx(num_segments=15, compressed=False))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        #
        # Set the options on the optimization variables
        # Note the use of explicit state units here since much of the ODE uses imperial units
        # and we prefer to solve this problem using metric units.
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                            duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='flight_dynamics.r_dot')

        # NOTE: added ref0
        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                        ref0=10, ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.h_dot')

        phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.v_dot')

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                        ref=1.0, defect_ref=1.0,
                        rate_source='flight_dynamics.gam_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='prop.m_dot')

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                        rate_continuity=True, rate_continuity_scaler=100.0,
                        rate2_continuity=False)

        phase.add_parameter('S', val=49.2386, units='m**2', opt=False, targets=['S'])
        phase.add_parameter('Isp', val=1600.0, units='s', opt=False, targets=['Isp'])
        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        #
        # Setup the boundary and path constraints
        #
        phase.add_boundary_constraint('h', loc='final', equals=20000, ref0=10)
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0)
        phase.add_boundary_constraint('gam', loc='final', equals=0.0)

        phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', ref=1.0)

        p.model.linear_solver = om.DirectSolver()

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 500

        p.set_val('traj.phase0.states:r', phase.interp('r', [0.0, 50000.0]))
        p.set_val('traj.phase0.states:h', phase.interp('h', [100.0, 20000.0]))
        p.set_val('traj.phase0.states:v', phase.interp('v', [135.964, 283.159]))
        p.set_val('traj.phase0.states:gam', phase.interp('gam', [0.0, 0.0]))
        p.set_val('traj.phase0.states:m', phase.interp('m', [19030.468, 10000.]))
        p.set_val('traj.phase0.controls:alpha', phase.interp('alpha', [0.0, 0.0]))

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, simulate=False)

        # Check that time matches to within 1% of an externally verified solution.
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 321.0, tolerance=0.02)

        # Verify that ODE output mach is added to the timeseries
        assert_near_equal(p.get_val('traj.phase0.timeseries.states:h')[-1], 20000.0, tolerance=1.0E-2)
        assert_near_equal(p.get_val('traj.phase0.timeseries.mach')[-1], 1.0, tolerance=1.0E-2)

    def test_state_ref0_force_override_nonlinear_bc(self):

        for tx in [dm.GaussLobatto, dm.Radau]:
            with self.subTest(tx):
                self.run_problem(tx)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
