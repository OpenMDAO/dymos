from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase, GaussLobatto, Radau
from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE


def ssto_earth(transcription='gauss-lobatto', num_seg=10, transcription_order=5,
               optimizer='SLSQP'):
    p = Problem(model=Group())
    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    else:
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

    t = {'gauss-lobatto': GaussLobatto(num_segments=num_seg, order=transcription_order),
         'radau-ps': Radau(num_segments=num_seg, order=transcription_order)}

    phase = Phase(ode_class=LaunchVehicleODE,
                  ode_init_kwargs={'central_body': 'earth'},
                  transcription=t[transcription])

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(10, 500))

    phase.set_state_options('x', fix_initial=True, ref=1.0E5, defect_ref=1.0)
    phase.set_state_options('y', fix_initial=True, ref=1.0E5, defect_ref=1.0)
    phase.set_state_options('vx', fix_initial=True, ref=1.0E3, defect_ref=1.0)
    phase.set_state_options('vy', fix_initial=True, ref=1.0E3, defect_ref=1.0)
    phase.set_state_options('m', fix_initial=True, ref=1.0E3, defect_ref=1.0)

    phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
    phase.add_boundary_constraint('vx', loc='final', equals=7796.6961)
    phase.add_boundary_constraint('vy', loc='final', equals=0)

    phase.add_control('theta', units='rad', lower=-1.57, upper=1.57)
    phase.add_design_parameter('thrust', units='N', opt=False, val=2100000.0)

    phase.add_objective('time', loc='final', scaler=0.01)

    p.model.linear_solver = DirectSolver()

    return p
