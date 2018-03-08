from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DenseJacobian, \
    CSCJacobian, CSRJacobian, DirectSolver

from dymos import Phase
from dymos.examples.ssto.launch_vehicle_linear_tangent_ode import LaunchVehicleLinearTangentODE

SHOW_PLOTS = True


def ssto_moon_linear_tangent(transcription='gauss-lobatto', num_seg=10, transcription_order=5,
                             top_level_jacobian='csc', optimizer='SLSQP', derivative_mode='rev'):

    p = Problem(model=Group())

    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    else:
        p.driver = ScipyOptimizeDriver()

    phase = Phase(transcription,
                  ode_class=LaunchVehicleLinearTangentODE,
                  ode_init_kwargs={'central_body': 'moon'},
                  num_segments=num_seg,
                  transcription_order=transcription_order)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 1000))

    phase.set_state_options('x', fix_initial=True, scaler=1.0E-5, lower=0)
    phase.set_state_options('y', fix_initial=True, scaler=1.0E-5, lower=0)
    phase.set_state_options('vx', fix_initial=True, scaler=1.0E-3, lower=0)
    phase.set_state_options('vy', fix_initial=True, scaler=1.0E-3)
    phase.set_state_options('m', fix_initial=True, scaler=1.0E-3)

    phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
    phase.add_boundary_constraint('vx', loc='final', equals=1627.0)
    phase.add_boundary_constraint('vy', loc='final', equals=0)

    phase.add_control('a_ctrl', units='1/s', dynamic=False, opt=True)
    phase.add_control('b_ctrl', units=None, dynamic=False, opt=True)
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

    p.setup(mode=derivative_mode, check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 500.0
    p['phase0.states:x'] = phase.interpolate(ys=[0, 350000.0], nodes='disc')
    p['phase0.states:y'] = phase.interpolate(ys=[0, 185000.0], nodes='disc')
    p['phase0.states:vx'] = phase.interpolate(ys=[0, 1627.0], nodes='disc')
    p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='disc')
    p['phase0.states:m'] = phase.interpolate(ys=[50000, 50000], nodes='disc')
    p['phase0.controls:a_ctrl'] = -0.01
    p['phase0.controls:b_ctrl'] = 3.0

    p.run_driver()

    if SHOW_PLOTS:  # pragma: no cover
        import matplotlib.pyplot as plt
        plt.figure(facecolor='white')
        plt.plot(phase.get_values('x'), phase.get_values('y'), 'bo')
        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.grid()

        fig = plt.figure(facecolor='white')
        fig.suptitle('results for flat_earth_without_aero')

        axarr = fig.add_subplot(2, 1, 1)
        axarr.plot(phase.get_values('time'),
                   np.degrees(phase.get_values('guidance.theta')), 'bo')
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

    return p


if __name__ == '__main__':
    ssto_moon_linear_tangent(transcription='gauss-lobatto', optimizer='SLSQP',
                             top_level_jacobian='csc', derivative_mode='rev')
