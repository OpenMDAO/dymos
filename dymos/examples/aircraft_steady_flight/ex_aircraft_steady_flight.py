from __future__ import division, print_function, absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from openmdao.api import IndepVarComp, ScipyOptimizeDriver

from dymos import Phase

from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE
from dymos.utils.lgl import lgl


def ex_aircraft_steady_flight(optimizer='SLSQP', transcription='gauss-lobatto'):
    p = Problem(model=Group())
    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.options['dynamic_simul_derivs'] = True
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 20
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings["Linesearch tolerance"] = 0.10
        p.driver.opt_settings['iSumm'] = 6
    if optimizer == 'SLSQP':
        p.driver.opt_settings['MAXIT'] = 50

    num_seg = 15
    seg_ends, _ = lgl(num_seg + 1)

    phase = Phase(transcription,
                  ode_class=AircraftODE,
                  num_segments=num_seg,
                  segment_ends=seg_ends,
                  transcription_order=3,
                  compressed=False)

    # Pass Reference Area from an external source
    assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
    assumptions.add_output('S', val=427.8, units='m**2')
    assumptions.add_output('mass_empty', val=1.0, units='kg')
    assumptions.add_output('mass_payload', val=1.0, units='kg')

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0),
                           duration_bounds=(300, 10000),
                           duration_ref=5600)

    phase.set_state_options('range', units='NM', fix_initial=True, fix_final=False, ref=1e-3,
                            defect_ref=1e-3)
    phase.set_state_options('mass_fuel', units='lbm', fix_initial=True, fix_final=True,
                            upper=1.5E5, lower=0.0, ref=1e2, defect_ref=1e2)
    phase.set_state_options('alt', units='kft', fix_initial=True, fix_final=True, lower=0.0,
                            upper=60, ref=1e-3, defect_ref=1e-3)

    phase.add_control('climb_rate', units='ft/min', opt=True, lower=-3000, upper=3000,
                      rate_continuity=True)

    phase.add_control('mach', units=None, opt=False)

    phase.add_input_parameter('S', units='m**2')
    phase.add_input_parameter('mass_empty', units='kg')
    phase.add_input_parameter('mass_payload', units='kg')

    phase.add_path_constraint('propulsion.tau', lower=0.01, upper=2.0)

    p.model.connect('assumptions.S', 'phase0.input_parameters:S')
    p.model.connect('assumptions.mass_empty', 'phase0.input_parameters:mass_empty')
    p.model.connect('assumptions.mass_payload', 'phase0.input_parameters:mass_payload')

    phase.add_objective('range', loc='final', ref=-1.0e-4)

    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup()

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 3600.0
    p['phase0.states:range'] = phase.interpolate(ys=(0, 724.0), nodes='state_input')
    p['phase0.states:mass_fuel'] = phase.interpolate(ys=(30000, 0), nodes='state_input')
    p['phase0.states:alt'][:] = 10.0

    p['phase0.controls:mach'][:] = 0.8

    p['assumptions.S'] = 427.8
    p['assumptions.mass_empty'] = 0.15E6
    p['assumptions.mass_payload'] = 84.02869 * 400

    p.run_driver()

    exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 500), record=True,
                             record_file='test_ex_aircraft_steady_flight_rec.db')

    plt.plot(phase.get_values('time', nodes='all'), phase.get_values('alt', nodes='all'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('alt'), 'b-')
    plt.suptitle('altitude vs time')
    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'),
             phase.get_values('climb_rate', nodes='all', units='ft/min'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('climb_rate', units='ft/min'), 'b-')
    plt.suptitle('climb rate vs time')
    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'), phase.get_values('mass_fuel', nodes='all'),
             'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('mass_fuel'), 'b-')
    plt.suptitle('fuel mass vs time')
    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'),
             phase.get_values('propulsion.dXdt:mass_fuel', nodes='all'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('propulsion.dXdt:mass_fuel'), 'b-')
    plt.suptitle('fuel mass flow rate vs time')
    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'), phase.get_values('mach', nodes='all'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('mach'), 'b-')
    plt.suptitle('mach vs time')
    plt.figure()
    plt.plot(phase.get_values('time', nodes='all'), phase.get_values('mach_rate', nodes='all'),
             'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('mach_rate'), 'b-')
    plt.suptitle('mach rate vs time')

    print('time')
    print(phase.get_values('time', nodes='all').T)

    print('alt')
    print(phase.get_values('alt', nodes='all').T)

    print('alt_rate')
    print(phase.get_values('climb_rate', nodes='all').T)

    print('climb_rate_rate')
    print(phase.get_values('climb_rate_rate', nodes='all').T)

    print('range')
    print(phase.get_values('range', nodes='all').T)

    print('flight path angle')
    print(phase.get_values('gam_comp.gam').T)

    print('true airspeed')
    print(phase.get_values('tas_comp.TAS', units='m/s').T)

    print('coef of lift')
    print(phase.get_values('aero.CL').T)

    print('coef of drag')
    print(phase.get_values('aero.CD').T)

    print('atmos density')
    print(phase.get_values('atmos.rho').T)

    print('alpha')
    print(phase.get_values('flight_equilibrium.alpha', units='rad').T)

    print('coef of thrust')
    print(phase.get_values('flight_equilibrium.CT').T)

    print('fuel flow rate')
    print(phase.get_values('propulsion.dXdt:mass_fuel').T)

    print('max_thrust')
    print(phase.get_values('propulsion.max_thrust', units='N').T)

    print('tau')
    print(phase.get_values('propulsion.tau').T)

    print('dynamic pressure')
    print(phase.get_values('q_comp.q', units='Pa').T)

    print('S')
    print(phase.get_values('S', units='m**2').T)

    plt.show()

    return p


if __name__ == '__main__':
    ex_aircraft_steady_flight(optimizer='SLSQP', transcription='radau-ps')
