from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, DenseJacobian, DirectSolver, \
    CSCJacobian, CSRJacobian

from openmdoc import Phase

from openmdoc.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE


def min_time_climb_problem(optimizer='SLSQP', num_seg=3, transcription_order=5,
                           transcription='gauss-lobatto', alpha_guess=False,
                           top_level_densejacobian=True, simul_derivs=False):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    # p.driver.options['simul_derivs'] = simul_derivs
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
        p.driver.opt_settings['Verify level'] = 3
        p.driver.opt_settings['Function precision'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.5

    phase = Phase(transcription,
                  ode_function=MinTimeClimbODE(),
                  num_segments=num_seg, transcription_order=transcription_order,
                  compressed=False)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(opt_initial=False, duration_bounds=(50, 400),
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
                      dynamic=True, rate_continuity=True)

    phase.add_control('S', val=49.2386, units='m**2', dynamic=False, opt=False)
    phase.add_control('Isp', val=1600.0, units='s', dynamic=False, opt=False)
    phase.add_control('throttle', val=1.0, dynamic=False, opt=False)

    phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
    phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
    phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

    phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
    phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)

    # Minimize time at the end of the phase
    phase.set_objective('time', loc='final', ref=100.0)

    if top_level_densejacobian:
        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

    p.setup(mode='fwd', check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 298.46902
    p['phase0.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='disc')
    p['phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='disc')
    p['phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='disc')
    p['phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='disc')
    p['phase0.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='disc')
    p['phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='all')

    if alpha_guess:
        interp_times = [0.0, 9.949, 19.9, 29.85, 39.8, 49.74, 59.69, 69.64, 79.59, 89.54, 99.49,
                        109.4, 119.4, 129.3, 139.3, 149.2, 159.2, 169.1, 179.1, 189.0, 199.0,
                        208.9, 218.9, 228.8, 238.8, 248.7, 258.7, 268.6, 278.6, 288.5, 298.5]

        interp_alphas = [0.07423, 0.03379, 0.01555, 0.01441, 0.02529, 0.0349, 0.02994, 0.02161,
                         0.02112, 0.02553, 0.0319, 0.03622, 0.03443, 0.03006, 0.0266, 0.0244,
                         0.02384, 0.02399, 0.02397, 0.02309, 0.0207, 0.01948, 0.0221, 0.02929,
                         0.04176, 0.05589, 0.06805, 0.06952, 0.05159, -0.002399, -0.1091]

        p['phase0.controls:alpha'] = phase.interpolate(xs=interp_times,
                                                       ys=interp_alphas, nodes='all')

    return p


if __name__ == '__main__':
    p = min_time_climb_problem(optimizer='SNOPT', num_seg=10, transcription_order=3)
    p.run_model()
    p.run_driver()
