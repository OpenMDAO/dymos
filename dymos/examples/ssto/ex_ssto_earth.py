from __future__ import print_function, division, absolute_import

import os

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DenseJacobian, \
    CSCJacobian, CSRJacobian, DirectSolver

from dymos import Phase
from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE

SHOW_PLOTS = True


def ssto_earth(transcription='gauss-lobatto', num_seg=10, transcription_order=5,
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
                  ode_class=LaunchVehicleODE,
                  ode_init_kwargs={'central_body': 'earth'},
                  num_segments=num_seg,
                  transcription_order=transcription_order)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 500))

    phase.set_state_options('x', fix_initial=True, scaler=1.0E-5)
    phase.set_state_options('y', fix_initial=True, scaler=1.0E-5)
    phase.set_state_options('vx', fix_initial=True, scaler=1.0E-3)
    phase.set_state_options('vy', fix_initial=True, scaler=1.0E-3)
    phase.set_state_options('m', fix_initial=True, scaler=1.0E-3)

    phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
    phase.add_boundary_constraint('vx', loc='final', equals=7796.6961)
    phase.add_boundary_constraint('vy', loc='final', equals=0)

    phase.add_control('theta', units='rad', dynamic=True, lower=-1.57, upper=1.57)
    phase.add_control('thrust', units='N', dynamic=False, opt=False, val=2100000.0)

    phase.add_objective('time', loc='final', scaler=0.01)

    if top_level_jacobian.lower() == 'csc':
        p.model.jacobian = CSCJacobian()
    elif top_level_jacobian.lower() == 'dense':
        p.model.jacobian = DenseJacobian()
    elif top_level_jacobian.lower() == 'csr':
        p.model.jacobian = CSRJacobian()

    p.model.linear_solver = DirectSolver()

    p.setup(mode=derivative_mode, check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 150.0
    p['phase0.states:x'] = phase.interpolate(ys=[0, 1.15E5], nodes='disc')
    p['phase0.states:y'] = phase.interpolate(ys=[0, 1.85E5], nodes='disc')
    p['phase0.states:vx'] = phase.interpolate(ys=[0, 7796.6961], nodes='disc')
    p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='disc')
    p['phase0.states:m'] = phase.interpolate(ys=[117000, 1163], nodes='disc')
    p['phase0.controls:theta'] = phase.interpolate(ys=[1.5, -0.76], nodes='all')

    p.run_model()

    p.run_driver()

    if SHOW_PLOTS:  # pragma: no cover
        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 100))

        import matplotlib.pyplot as plt
        plt.figure(facecolor='white')
        plt.plot(phase.get_values('x'), phase.get_values('y'), 'bo', label='solution')
        plt.plot(exp_out.get_values('x'), exp_out.get_values('y'), 'r-', label='simulated')
        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.legend()
        plt.grid()

        fig = plt.figure(facecolor='white')
        fig.suptitle('results for flat_earth_without_aero')

        axarr = fig.add_subplot(2, 1, 1)
        axarr.plot(phase.get_values('time'),
                   phase.get_values('theta', units='deg'), 'bo', label='solution')
        axarr.plot(exp_out.get_values('time'),
                   exp_out.get_values('theta', units='deg'), 'b-', label='simulated')
        axarr.set_ylabel(r'$\theta$, deg')
        axarr.axes.get_xaxis().set_visible(False)

        axarr = fig.add_subplot(2, 1, 2)

        axarr.plot(phase.get_values('time'),
                   phase.get_values('vx'), 'bo', label='$v_x$ solution')
        axarr.plot(exp_out.get_values('time'),
                   exp_out.get_values('vx'), 'b-', label='$v_x$ simulated')

        axarr.plot(phase.get_values('time'),
                   phase.get_values('vy'), 'ro', label='$v_y$ solution')
        axarr.plot(exp_out.get_values('time'),
                   exp_out.get_values('vy'), 'r-', label='$v_y$ simulated')

        axarr.set_xlabel('time, s')
        axarr.set_ylabel('velocity, m/s')
        axarr.legend(loc='best')
        plt.show()

        os.remove('phase0_sim.db')

    return p


if __name__ == '__main__':
    ssto_earth(transcription='gauss-lobatto', top_level_jacobian='csc',
               derivative_mode='fwd')
