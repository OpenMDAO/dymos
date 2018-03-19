from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DenseJacobian, \
    CSCJacobian, CSRJacobian, DirectSolver

from dymos import Phase
from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE


def ssto_moon(transcription='gauss-lobatto', num_seg=10, optimizer='SLSQP',
              top_level_jacobian='csc', transcription_order=5,
              glm_formulation='solver-based', glm_integrator='GaussLegendre4'):

    p = Problem(model=Group())

    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    else:
        p.driver = ScipyOptimizeDriver()

    kwargs = {}
    if transcription == 'gauss-lobatto' or transcription == 'radau-ps':
        kwargs['transcription_order'] = 3
    else:
        kwargs['formulation'] = glm_formulation
        kwargs['method_name'] = glm_integrator

    phase = Phase(transcription,
                  ode_class=LaunchVehicleODE,
                  ode_init_kwargs={'central_body': 'moon'},
                  num_segments=num_seg,
                  **kwargs)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 1000))

    if transcription != 'glm':
        phase.set_state_options('x', fix_initial=True, scaler=1.0E-5, lower=0)
        phase.set_state_options('y', fix_initial=True, scaler=1.0E-5, lower=0)
        phase.set_state_options('vx', fix_initial=True, scaler=1.0E-3, lower=0)
        phase.set_state_options('vy', fix_initial=True, scaler=1.0E-3)
        phase.set_state_options('m', fix_initial=True, scaler=1.0E-3)
    else:
        phase.set_state_options('x', fix_initial=True, fix_final=False, scaler=1.0E-5, lower=0)
        phase.set_state_options('y', fix_initial=True, fix_final=False, scaler=1.0E-5, lower=0)
        phase.set_state_options('vx', fix_initial=True, fix_final=False, scaler=1.0E-3, lower=0)
        phase.set_state_options('vy', fix_initial=True, fix_final=False, scaler=1.0E-3)
        phase.set_state_options('m', fix_initial=True, fix_final=False, scaler=1.0E-3)

        phase.add_boundary_constraint('x', loc='initial', equals=0.)
        phase.add_boundary_constraint('y', loc='initial', equals=0.)
        phase.add_boundary_constraint('vx', loc='initial', equals=0.)
        phase.add_boundary_constraint('vy', loc='initial', equals=1.0E-6)
        phase.add_boundary_constraint('m', loc='initial', equals=50000.)

    phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
    phase.add_boundary_constraint('vx', loc='final', equals=1627.0)
    phase.add_boundary_constraint('vy', loc='final', equals=0)

    if transcription == 'radau-ps':
        # This constraint is necessary using the Radau-Pseudospectral method since the
        # last value of the control does not impact the collocation defects.
        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0)

    phase.add_control('theta', units='rad', dynamic=True, lower=-1.57, upper=1.57)
    phase.add_control('thrust', units='N', dynamic=False, opt=False, val=3.0 * 50000.0 * 1.61544)
    phase.add_control('Isp', units='s', dynamic=False, opt=False, val=1.0E6)

    phase.add_objective('time', index=-1, scaler=0.01)

    if top_level_jacobian.lower() == 'csc':
        p.model.jacobian = CSCJacobian()
    elif top_level_jacobian.lower() == 'dense':
        p.model.jacobian = DenseJacobian()
    elif top_level_jacobian.lower() == 'csr':
        p.model.jacobian = CSRJacobian()

    p.model.linear_solver = DirectSolver()

    return p


if __name__ == "__main__":

    # pragma: no cover
    import matplotlib.pyplot as plt

    p = ssto_moon('gauss-lobatto', num_seg=10, transcription_order=5, top_level_jacobian='csc')

    p.setup(mode='rev', check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 500.0

    phase = p.model.phase0

    p['phase0.states:x'] = phase.interpolate(ys=[0, 350000.0], nodes='disc')
    p['phase0.states:y'] = phase.interpolate(ys=[0, 185000.0], nodes='disc')
    p['phase0.states:vx'] = phase.interpolate(ys=[0, 1627.0], nodes='disc')
    p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='disc')
    p['phase0.states:m'] = phase.interpolate(ys=[50000, 50000], nodes='disc')
    p['phase0.controls:theta'] = phase.interpolate(ys=[1.5, -0.76], nodes='all')

    p.run_driver()

    plt.figure(facecolor='white')
    plt.plot(phase.get_values('x'), phase.get_values('y'), 'bo')
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.grid()

    fig = plt.figure(facecolor='white')
    fig.suptitle('results for flat_earth_without_aero')

    axarr = fig.add_subplot(2, 1, 1)
    axarr.plot(phase.get_values('time'),
               np.degrees(phase.get_values('theta')), 'bo')
    axarr.set_ylabel(r'$\theta$, deg')
    axarr.axes.get_xaxis().set_visible(False)

    axarr = fig.add_subplot(2, 1, 2)

    axarr.plot(phase.get_values('time'),
               np.degrees(phase.get_values('vx')), 'bo', label='$v_x$')
    axarr.plot(phase.get_values('time'),
               np.degrees(phase.get_values('vy')), 'ro', label='$v_y$')
    axarr.set_xlabel('time, s')
    axarr.set_ylabel('velocity, m/s')
    axarr.legend(loc='best')
    plt.show()
