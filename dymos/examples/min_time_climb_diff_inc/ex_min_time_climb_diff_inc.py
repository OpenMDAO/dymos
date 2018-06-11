from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from dymos import Phase

from dymos.examples.min_time_climb_diff_inc.min_time_climb_diff_inc_ode \
    import MinTimeClimbDiffIncODE

SHOW_PLOTS = False


def min_time_climb_diff_inc(optimizer='SLSQP', num_seg=3, transcription='gauss-lobatto',
                            transcription_order=5, top_level_jacobian='csc'):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer

    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['Iterations limit'] = 100000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        # p.driver.opt_settings['Verify level'] = 3
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.1

    phase = Phase(transcription,
                  ode_class=MinTimeClimbDiffIncODE,
                  num_segments=num_seg,
                  compressed=False,
                  transcription_order=transcription_order)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(opt_initial=False, duration_bounds=(50, 400),
                           duration_ref=100.0)

    phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('h', fix_initial=True, fix_final=True, lower=0, upper=20000.0,
                            scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

    phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                            scaler=1.0E-3, defect_scaler=1.0E-3)

    phase.add_control('v', rate_param='v_rate', val=150.0, units='m/s', dynamic=True, opt=True,
                      fix_initial=True, rate_continuity=True, rate2_continuity=True)
    phase.add_control('gam', rate_param='gam_rate', val=0.0, units='deg', dynamic=True, opt=True,
                      fix_initial=True, fix_final=True, rate_continuity=True, rate2_continuity=True)
    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    phase.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)

    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
    phase.add_path_constraint(name='alpha', lower=-14, upper=14)
    phase.add_path_constraint(name='throttle', lower=0.1, upper=1.0)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', ref=1.0)

    if transcription != 'glm':
        p.driver.options['dynamic_simul_derivs'] = True

        if top_level_jacobian.lower() == 'csc':
            p.model.jacobian = CSCJacobian()
        elif top_level_jacobian.lower() == 'dense':
            p.model.jacobian = DenseJacobian()
        elif top_level_jacobian.lower() == 'csr':
            p.model.jacobian = CSRJacobian()

        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd', check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 298.46902

        p['phase0.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='state_disc')
        p['phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_disc')
        p['phase0.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='state_disc')
        p['phase0.controls:v'] = phase.interpolate(ys=[135.964, 150.0], nodes='control_disc')
        p['phase0.controls:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_disc')

    p.run_driver()

    if SHOW_PLOTS:
        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 100))

        import matplotlib.pyplot as plt
        plt.plot(phase.get_values('time'), phase.get_values('h'), 'ro')
        plt.plot(exp_out.get_values('time'), exp_out.get_values('h'), 'b-')
        plt.xlabel('time (s)')
        plt.ylabel('altitude (m)')

        plt.figure()
        plt.plot(phase.get_values('v'), phase.get_values('h'), 'ro')
        plt.plot(exp_out.get_values('v'), exp_out.get_values('h'), 'b-')
        plt.xlabel('airspeed (m/s)')
        plt.ylabel('altitude (m)')

        plt.figure()
        plt.plot(phase.get_values('time'), phase.get_values('alpha'), 'ro')
        plt.plot(exp_out.get_values('time'), exp_out.get_values('alpha'), 'b-')
        plt.xlabel('time (s)')
        plt.ylabel('alpha (rad)')

        plt.show()

    return p


if __name__ == '__main__':
    SHOW_PLOTS = True
    p = min_time_climb_diff_inc(
        optimizer='SNOPT', num_seg=5, transcription='gauss-lobatto', transcription_order=7)
