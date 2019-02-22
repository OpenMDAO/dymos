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


def ex_aircraft_steady_flight(optimizer='SLSQP', transcription='gauss-lobatto',
                              solve_segments=False, show_plots=False,
                              use_boundary_constraints=False, compressed=False):
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
                  compressed=compressed)

    # Pass Reference Area from an external source
    assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
    assumptions.add_output('S', val=427.8, units='m**2')
    assumptions.add_output('mass_empty', val=1.0, units='kg')
    assumptions.add_output('mass_payload', val=1.0, units='kg')

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0),
                           duration_bounds=(300, 10000),
                           duration_ref=5600)

    fix_final = True
    if use_boundary_constraints:
        fix_final = False
        phase.add_boundary_constraint('mass_fuel', loc='final', units='lbm',
                                      equals=1e-3, linear=False)
        phase.add_boundary_constraint('alt', loc='final', units='kft', equals=10.0, linear=False)

    phase.set_state_options('range', units='NM', fix_initial=True, fix_final=False, ref=1e-3,
                            defect_ref=1e-3, lower=0, upper=2000, solve_segments=solve_segments)
    phase.set_state_options('mass_fuel', units='lbm', fix_initial=True, fix_final=fix_final,
                            upper=1.5E5, lower=0.0, ref=1e2, defect_ref=1e2,
                            solve_segments=solve_segments)
    phase.set_state_options('alt', units='kft', fix_initial=True, fix_final=fix_final, lower=0.0,
                            upper=60, ref=1e-3, defect_ref=1e-3,
                            solve_segments=solve_segments)

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

    p.setup()

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 3600.0
    p['phase0.states:range'][:] = phase.interpolate(ys=(0, 724.0), nodes='state_input')
    p['phase0.states:mass_fuel'][:] = phase.interpolate(ys=(30000, 1e-3), nodes='state_input')
    p['phase0.states:alt'][:] = 10.0

    p['phase0.controls:mach'][:] = 0.8

    p['assumptions.S'] = 427.8
    p['assumptions.mass_empty'] = 0.15E6
    p['assumptions.mass_payload'] = 84.02869 * 400

    p.run_driver()

    if show_plots:
        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 500), record=True,
                                 record_file='test_ex_aircraft_steady_flight_rec.db')

        t_imp = p.get_val('phase0.timeseries.time')
        t_exp = exp_out.get_val('phase0.timeseries.time')

        alt_imp = p.get_val('phase0.timeseries.states:alt')
        alt_exp = exp_out.get_val('phase0.timeseries.states:alt')

        climb_rate_imp = p.get_val('phase0.timeseries.controls:climb_rate', units='ft/min')
        climb_rate_exp = exp_out.get_val('phase0.timeseries.controls:climb_rate', units='ft/min')

        mass_fuel_imp = p.get_val('phase0.timeseries.states:mass_fuel', units='kg')
        mass_fuel_exp = exp_out.get_val('phase0.timeseries.states:mass_fuel', units='kg')

        plt.plot(t_imp, alt_imp, 'ro')
        plt.plot(t_exp, alt_exp, 'b-')
        plt.suptitle('altitude vs time')

        plt.figure()
        plt.plot(t_imp, climb_rate_imp, 'ro')
        plt.plot(t_exp, climb_rate_exp, 'b-')
        plt.suptitle('climb rate vs time')

        plt.figure()
        plt.plot(t_imp, mass_fuel_imp, 'ro')
        plt.plot(t_exp, mass_fuel_exp, 'b-')
        plt.suptitle('fuel mass vs time')

        plt.show()

    return p


if __name__ == '__main__':
    import time

    st = time.time()

    ex_aircraft_steady_flight(optimizer='SNOPT', transcription='radau-ps',
                              compressed=True, show_plots=False)

    # ex_aircraft_steady_flight(optimizer='SNOPT', transcription='gauss-lobatto',
    #                           solve_segments=True, use_boundary_constraints=True,
    #                           compressed=False, show_plots=False)

    print('time: ', time.time() - st)
