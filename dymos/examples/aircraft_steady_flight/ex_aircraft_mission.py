from __future__ import division, print_function, absolute_import

import numpy as np

from openmdao.api import Problem, Group, ScipyOptimizeDriver, pyOptSparseDriver, DirectSolver, \
    CSCJacobian, IndepVarComp, SqliteRecorder

from dymos import Phase

from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE
# from dymos.examples.aircraft_steady_flight.cost_comp import CostComp

# Demonstrates:
# 1. Externally sourced controls (S)
# 2. Externally computed objective (cost)

# TODO:
#


def ex_aircraft_mission(transcription='radau-ps', num_seg=10, transcription_order=3,
                        seg_ends=None, optimizer='SNOPT', compressed=True):

        p = Problem(model=Group())
        if optimizer == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.opt_settings['Major iterations limit'] = 100
            # p.driver.opt_settings['Major step limit'] = 0.05
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            # p.driver.opt_settings["Linesearch tolerance"] = 0.10
            p.driver.opt_settings['iSumm'] = 6
            # p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = ScipyOptimizeDriver()
            p.driver.options['dynamic_simul_derivs'] = True

        rec = SqliteRecorder("aircraft_mission_results.db")
        p.driver.add_recorder(rec)
        p.driver.recording_options['record_desvars'] = True

        phase = Phase(transcription,
                      ode_class=AircraftODE,
                      num_segments=num_seg,
                      transcription_order=transcription_order, segment_ends=seg_ends,
                      compressed=compressed)

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0),
                               duration_bounds=(10, 7200),
                               duration_ref=3600)

        phase.set_state_options('range', units='km', fix_initial=True, fix_final=False, scaler=0.01,
                                defect_scaler=0.01)
        phase.set_state_options('mass_fuel', fix_initial=False, fix_final=True, upper=25000.0,
                                lower=0.0, scaler=1.0E-4, defect_scaler=1.0E-2)

        phase.add_control('alt', units='m', dynamic=True, opt=True, lower=0.0, upper=15000.0,
                          rate_param='climb_rate', rate_continuity=True, rate2_continuity=False,
                          fix_initial=True, fix_final=True)

        phase.add_control('mach', units=None, dynamic=False, opt=False,
                          lower=0.8, upper=0.8, ref=1.0)

        phase.add_control('S', units='m**2', dynamic=False, opt=False)
        phase.add_control('mass_empty', units='kg', dynamic=False, opt=False)
        phase.add_control('mass_payload', units='kg', dynamic=False, opt=False)

        phase.add_path_constraint('flight_equilibrium.alpha', lower=-14, upper=14, units='deg')
        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)
        phase.add_path_constraint('alt_rate', lower=-2000, upper=2000, ref=2000.0, units='ft/min')
        # phase.add_path_constraint('alt_rate2', lower=-0.05, upper=0.05, ref=0.01, units='m/s**2')

        phase.add_boundary_constraint('range', loc='final', equals=1296.4, ref=100.0, units='km')
        # phase.add_boundary_constraint('mass', loc='final', lower=200000.0, ref=100000, units='kg')

        p.model.connect('assumptions.S', 'phase0.controls:S')
        p.model.connect('assumptions.mass_empty', 'phase0.controls:mass_empty')
        p.model.connect('assumptions.mass_payload', 'phase0.controls:mass_payload')

        # p.model.add_subsystem('cost_comp', subsys=CostComp())
        #
        # p.model.connect('phase0.t_duration', 'cost_comp.tof')
        # p.model.connect('phase0.states:mass_fuel', 'cost_comp.initial_mass_fuel', src_indices=[0])

        # phase.add_objective('range', loc='final', ref=-100)
        # p.model.add_objective('mass_fuel', loc='initial', ref=1E4)
        phase.add_objective('mass_fuel', loc='initial', ref=1E3)

        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.515132 * 3600.0
        p['phase0.states:range'] = phase.interpolate(ys=(0, 1296.4), nodes='state_disc')
        p['phase0.states:mass_fuel'] = phase.interpolate(ys=(12236.0, 0), nodes='state_disc')
        p['phase0.controls:mach'] = 0.7
        # p['phase0.controls:TAS'][0] = p['phase0.controls:TAS'][-1] = 100.0
        p['phase0.controls:alt'] = phase.interpolate(ys=(0, 0), nodes='control_disc')

        p['assumptions.S'] = 427.8
        p['assumptions.mass_empty'] = 0.15E6
        p['assumptions.mass_payload'] = 84.02869 * 400

        return p


if __name__ == '__main__':

    from dymos.utils.lgl import lgl

    num_seg = 30
    seg_ends, _ = lgl(num_seg+1)
    p = ex_aircraft_mission(transcription='gauss-lobatto', num_seg=num_seg, seg_ends=seg_ends,
                            transcription_order=3)

    # from openmdao.api import view_model
    #
    # view_model(p.model)
    #
    # exit(0)
    # from openmdao.api import view_model
    # view_model(p.model)

    p.run_driver()

    exp_out = p.model.phase0.simulate(times=np.linspace(0,
                                                        p.model.phase0.get_values('time')[-1], 500))

    np.set_printoptions(linewidth=1024)
    print(p['phase0.continuity_constraint.defect_control_rates:alt_rate'])
    print()
    print(p['phase0.control_rate_comp.control_rates:alt_rate'].T)

    segend_idxs = p.model.phase0.grid_data.subset_node_indices['segment_ends']
    print(p['phase0.control_rate_comp.control_rates:alt_rate'][segend_idxs, :])

    print()
    print(p['phase0.continuity_constraint.control_rates:alt_rate'].T)
    print()
    print(p['phase0.control_rate_comp.dt_dstau'])

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(p.model.phase0.get_values('range', units='km'),
             p.model.phase0.get_values('alt', units='km'),
             'ro')
    plt.plot(exp_out.get_values('range', units='km'),
             exp_out.get_values('alt', units='km'),
             'b-')

    # plt.figure()
    # plt.plot(p.model.phase0.get_values('alt', units='km'),
    #          p.model.phase0.get_values('atmos.rho', units='kg/m**3'),
    #          'ro')
    # plt.plot(exp_out.get_values('alt', units='km'),
    #          exp_out.get_values('atmos.rho', units='kg/m**3'),
    #          'b-')

    # plt.figure()
    # plt.plot(p.model.phase0.get_values('range', units='km'),
    #          p.model.phase0.get_values('q_comp.q', units='Pa'),
    #          'ro')
    # plt.plot(exp_out.get_values('range', units='km'),
    #          exp_out.get_values('q_comp.q', units='Pa'),
    #          'b-')

    for var, units in [('mass_fuel', 'kg'), ('flight_equilibrium.alpha', 'deg'),
                       ('propulsion.tau', None), ('alt', 'km'), ('alt_rate', 'm/s'),
                       ('alt_rate2', 'm/s**2'), ('tas_comp.TAS', 'm/s'),
                       ('propulsion.dXdt:mass_fuel', 'kg/s'), ('mass_fuel', 'kg')]:
        plt.figure()
        print(var)
        plt.plot(p.model.phase0.get_values('time'),
                 p.model.phase0.get_values(var, units=units),
                 'ro')
        plt.plot(exp_out.get_values('time'),
                 exp_out.get_values(var, units=units),
                 'b-')
        plt.suptitle('{0} ({1})'.format(var, units))

    plt.show()
