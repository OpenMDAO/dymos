"""
minimum time reorientation of an asymmetric spacecraft

This example serves as a test to verify that Dymos works for ODEs in which the states, controls,
and input/design parameters are not scalars.

references
----------
.. [1] betts, john t., practical methods for optimal control and estimation using nonlinear
       programming, p. 299, 2010.
.. [2] fleming, sekhavat, and ross, minimum-time reorientation of an assymmetric rigid body,
       aiaa 2008-7012.
"""
from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import Problem, Group, DirectSolver, pyOptSparseDriver
from dymos import Phase
from dymos.examples.spacecraft_reorientation.spacecraft_reorientation_ode import \
    SpacecraftReorientationODE


def spacecraft_reorientation(optimizer='SNOPT', num_seg=20, transcription='radau-ps',
                             compressed=True, transcription_order=3):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.options['dynamic_simul_derivs'] = True

    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Function precision'] = 1.0E-12
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        p.driver.opt_settings['Major step limit'] = 0.5
        p.driver.opt_settings['Verify level'] = 3

    phase = Phase(transcription,
                  ode_class=SpacecraftReorientationODE,
                  num_segments=num_seg,
                  compressed=compressed,
                  transcription_order=transcription_order)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(5, 30),
                           duration_ref=30.0)

    phase.set_state_options('q', fix_initial=True, fix_final=True, defect_ref=1.0)

    phase.set_state_options('w', fix_initial=True, fix_final=False, ref=0.01, defect_ref=1.0)

    phase.add_control('u', fix_initial=True, lower=-100.0, upper=100.0, ref0=-50, ref=50.0)

    phase.add_design_parameter('I', units='kg*m**2', opt=False)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final')

    p.model.options['assembled_jac_type'] = 'csc'
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(check=True, force_alloc_complex=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 40.0

    dangle = np.radians(150./2)

    q_0 = [0, 0, 0, 1]
    q_f = [np.sin(dangle), 0, 0, np.cos(dangle)]

    w_0 = w_f = [0, 0, 0]

    u_0 = [50, -50, 50]
    u_f = [-50, 50, 0]

    p['phase0.states:q'] = phase.interpolate(ys=[q_0, q_f], nodes='state_input')
    p['phase0.states:w'] = phase.interpolate(ys=[w_0, w_f], nodes='state_input')
    p['phase0.controls:u'] = phase.interpolate(ys=[u_0, u_f], nodes='control_input')
    p['phase0.design_parameters:I'][:] = [5621., 4547., 2364.]

    p.run_model()
    p.run_driver()

    return p


if __name__ == '__main__':
    p = spacecraft_reorientation('SNOPT', num_seg=1, compressed=False,
                                 transcription='radau-ps', transcription_order=19)

    phase = p.model.phase0
    exp_out = phase.simulate()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(phase.get_values('time'), phase.get_values('q'), marker='o',
             label=['q0', 'q1', 'q2', 'q3'])
    plt.plot(exp_out.get_values('time'), exp_out.get_values('q'),
             label=['q0', 'q1', 'q2', 'q3'])
    plt.legend()

    plt.figure()
    plt.plot(phase.get_values('time'), phase.get_values('w'), 'ro', label='w')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('w'), 'b-', label='w')
    plt.legend()

    plt.figure()
    plt.plot(phase.get_values('time'), phase.get_values('u'), 'ro', label='w')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('u'), 'b-', label='w')
    plt.legend()

    plt.show()
