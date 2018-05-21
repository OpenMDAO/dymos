from __future__ import division, print_function, absolute_import

import numpy as np

from openmdao.api import Problem, Group, ScipyOptimizeDriver, pyOptSparseDriver, DirectSolver, \
    CSCJacobian, IndepVarComp

from dymos import Phase

from dymos.examples.aircraft.aircraft_ode import AircraftODE
from dymos.examples.aircraft.cost_comp import CostComp

# Demonstrates:
# 1. Externally sourced controls (S)
# 2. Externally computed objective (cost)

# TODO:
#


def ex_aircraft_mission(transcription='radau-ps', num_seg=10, transcription_order=3,
                        optimizer='SNOPT', compressed=True):

        p = Problem(model=Group())
        if optimizer == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.options['dynamic_simul_derivs_repeats'] = 5
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major step limit'] = 0.05
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            p.driver.opt_settings["Linesearch tolerance"] = 0.10
            p.driver.opt_settings['iSumm'] = 6
            # p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = ScipyOptimizeDriver()
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.options['dynamic_simul_derivs_repeats'] = 5

        phase = Phase(transcription,
                      ode_class=AircraftODE,
                      num_segments=num_seg,
                      transcription_order=transcription_order,
                      compressed=compressed)

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0),
                               duration_bounds=(100, 7200),
                               duration_ref=3600)

        phase.set_state_options('range', units='km', fix_initial=True, fix_final=False, scaler=0.01)
        phase.set_state_options('mass_fuel', fix_initial=False, upper=20000.0, lower=0.0,
                                scaler=1.0E-4, defect_scaler=10.0)

        phase.add_control('alt', units='km', dynamic=True, opt=True, lower=0.0, upper=11.0,
                          rate_param='climb_rate', rate_continuity=True,
                          rate2_param='climb_rate2', rate2_continuity=True, ref=1.0)

        phase.add_control('TAS', units='m/s', dynamic=True, opt=True, lower=250.0, upper=250.0,
                          rate_param='TAS_rate', rate_continuity=True, rate2_continuity=True,
                          ref=100.0)

        phase.add_control('S', units='m**2', dynamic=False, opt=False)
        phase.add_control('mass_empty', units='kg', dynamic=False, opt=False)
        phase.add_control('mass_payload', units='kg', dynamic=False, opt=False)

        phase.add_path_constraint('flight_equilibrium.alpha', lower=-14, upper=14, units='deg')
        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)
        # phase.add_path_constraint('TAS_rate', lower=-0.1*9.80665, upper=0.1*9.80665,
        # units='m/s**2')
        # phase.add_path_constraint('TAS_rate2', lower=-0.01, upper=0.01, units='m/s**3')
        # phase.add_path_constraint('alt_rate', lower=-3000, upper=3000, units='ft/min')
        phase.add_boundary_constraint('alt', loc='initial', equals=10.0)
        phase.add_boundary_constraint('alt', loc='final', equals=10.0)
        # phase.add_boundary_constraint('TAS', loc='initial', equals=200.0)
        # phase.add_boundary_constraint('TAS', loc='final', equals=200.0)
        phase.add_boundary_constraint('range', loc='final', equals=1296.4, ref=100.0, units='km')
        # phase.add_boundary_constraint('mass', loc='final', lower=200000.0, ref=100000, units='kg')

        p.model.connect('assumptions.S', 'phase0.controls:S')

        p.model.add_subsystem('cost_comp', subsys=CostComp())

        p.model.connect('phase0.t_duration', 'cost_comp.tof')
        p.model.connect('phase0.states:mass_fuel', 'cost_comp.initial_mass_fuel', src_indices=[0])

        # phase.add_objective('range', loc='final', ref=-100)
        # p.model.add_objective('mass_fuel', loc='initial', ref=1E4)
        phase.add_objective('mass_fuel', loc='initial', ref=1E4)

        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.515132 * 3600.0
        p['phase0.states:range'] = phase.interpolate(ys=(0, 1296.4), nodes='state_disc')
        p['phase0.states:mass_fuel'] = phase.interpolate(ys=(12236.0, 0), nodes='state_disc')
        p['phase0.controls:TAS'] = phase.interpolate(ys=(250, 250), nodes='control_disc')
        # p['phase0.controls:TAS'][0] = p['phase0.controls:TAS'][-1] = 100.0
        p['phase0.controls:alt'] = phase.interpolate(ys=(9.144, 9.144), nodes='control_disc')
        # p['phase0.controls:alt'][0] = p['phase0.controls:alt'][-1] = 0.0
        p['phase0.controls:S'] = 427.8
        p['phase0.controls:mass_empty'] = 0.15E6
        p['phase0.controls:mass_payload'] = 84.02869 * 400

        return p


if __name__ == '__main__':

    p = ex_aircraft_mission(transcription='gauss-lobatto', num_seg=12, transcription_order=5)

    # from openmdao.api import view_model
    #
    # view_model(p.model)
    #
    # exit(0)

    p.run_driver()

    exp_out = p.model.phase0.simulate(times='all')

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(p.model.phase0.get_values('range', units='km'),
             p.model.phase0.get_values('alt', units='km'),
             'ro')
    plt.plot(exp_out.get_values('range', units='km'),
             exp_out.get_values('alt', units='km'),
             'b-')

    plt.figure()
    plt.plot(p.model.phase0.get_values('alt', units='km'),
             p.model.phase0.get_values('atmos.rho', units='kg/m**3'),
             'ro')
    plt.plot(exp_out.get_values('alt', units='km'),
             exp_out.get_values('atmos.rho', units='kg/m**3'),
             'b-')

    plt.figure()
    plt.plot(p.model.phase0.get_values('range', units='km'),
             p.model.phase0.get_values('q_comp.q', units='Pa'),
             'ro')
    plt.plot(exp_out.get_values('range', units='km'),
             exp_out.get_values('q_comp.q', units='Pa'),
             'b-')

    for var, units in [('mass_fuel', 'kg'), ('flight_equilibrium.alpha', 'deg'),
                       ('propulsion.tau', None), ('TAS_rate', 'm/s**2'), ('alt', 'km'),
                       ('alt_rate', 'ft/min'), ('mach_comp.mach', None),
                       ('propulsion.dXdt:mass_fuel', 'kg/s')]:
        plt.figure()
        plt.plot(p.model.phase0.get_values('time'),
                 p.model.phase0.get_values(var, units=units),
                 'ro')
        plt.plot(exp_out.get_values('time'),
                 exp_out.get_values(var, units=units),
                 'b-')
        plt.suptitle('{0} ({1})'.format(var, units))

    plt.show()
