from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver

from dymos import Phase

from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

import matplotlib
matplotlib.use('Agg')

SHOW_PLOTS = True


def min_time_climb(optimizer='SLSQP', num_seg=3, transcription='gauss-lobatto',
                   transcription_order=3, num_steps=10, top_level_jacobian='csc', simul_derivs=True,
                   force_alloc_complex=False):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if simul_derivs:
        p.driver.options['dynamic_simul_derivs'] = True

    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.5
        # p.driver.opt_settings['Verify level'] = 3

    phase = Phase(transcription,
                  ode_class=MinTimeClimbODE,
                  num_segments=num_seg,
                  compressed=False,
                  transcription_order=transcription_order)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                           duration_ref=100.0)

    phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

    phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('v', fix_initial=True, lower=10.0,
                            scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

    phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                            ref=1.0, defect_scaler=1.0, units='rad')

    phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                      continuity=True, rate_continuity=True, rate2_continuity=True)

    phase.add_design_parameter('S', val=49.2386, units='m**2', opt=False)
    phase.add_design_parameter('Isp', val=1600.0, units='s', opt=False)
    phase.add_design_parameter('throttle', val=1.0, opt=False)

    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
    phase.add_path_constraint(name='alpha', lower=-8, upper=8)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final')

    p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 300.0

    p['phase0.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='state_input')
    p['phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
    p['phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
    p['phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
    p['phase0.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='state_input')
    p['phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

    p.run_driver()

    if SHOW_PLOTS:
        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 100))

        t_sol = p.get_val('phase0.timeseries.time')
        t_exp = exp_out.get_val('phase0.timeseries.time')

        h_sol = p.get_val('phase0.timeseries.states:h')
        h_exp = exp_out.get_val('phase0.timeseries.states:h')

        v_sol = p.get_val('phase0.timeseries.states:v')
        v_exp = exp_out.get_val('phase0.timeseries.states:v')

        alpha_sol = p.get_val('phase0.timeseries.controls:alpha')
        alpha_exp = exp_out.get_val('phase0.timeseries.controls:alpha')

        import matplotlib.pyplot as plt
        plt.plot(t_sol, h_sol, 'ro')
        plt.plot(t_exp, h_exp, 'b-')
        plt.xlabel('time (s)')
        plt.ylabel('altitude (m)')

        plt.figure()
        plt.plot(v_sol, h_sol, 'ro')
        plt.plot(v_exp, h_exp, 'b-')
        plt.xlabel('airspeed (m/s)')
        plt.ylabel('altitude (m)')

        plt.figure()
        plt.plot(t_sol, alpha_sol, 'ro')
        plt.plot(t_exp, alpha_exp, 'b-')
        plt.xlabel('time (s)')
        plt.ylabel('alpha (rad)')

        # plt.figure()
        # plt.plot(phase.get_values('time'), phase.get_values('prop.thrust', units='lbf'), 'ro')
        # plt.plot(exp_out.get_values('time'), exp_out.get_values('prop.thrust', units='lbf'), 'b-')
        # plt.xlabel('time (s)')
        # plt.ylabel('thrust (lbf)')

        plt.show()

    return p


if __name__ == '__main__':
    SHOW_PLOTS = True
    p = min_time_climb(
        optimizer='SNOPT', num_seg=10, transcription='radau-ps', transcription_order=5,
        force_alloc_complex=False, simul_derivs=True)
