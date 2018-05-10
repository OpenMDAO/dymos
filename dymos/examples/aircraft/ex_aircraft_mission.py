from __future__ import division, print_function, absolute_import

import numpy as np

from openmdao.api import Problem, Group, ScipyOptimizeDriver, pyOptSparseDriver, DirectSolver, \
    CSCJacobian

from dymos import Phase

from dymos.examples.aircraft.aircraft_ode import AircraftODE


def ex_aircraft_mission(transcription='radau-ps', num_seg=10, transcription_order=3,
                        optimizer='SNOPT', compressed=True):

        p = Problem(model=Group())
        if optimizer == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.options['dynamic_simul_derivs_repeats'] = 5
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = ScipyOptimizeDriver()
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.options['dynamic_simul_derivs_repeats'] = 5

        phase = Phase(transcription,
                      ode_class=AircraftODE,
                      num_segments=num_seg,
                      transcription_order=transcription_order,
                      compressed=compressed)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(100, 100))

        phase.set_state_options('range', units='km', fix_initial=True, scaler=1.0E-5)
        phase.set_state_options('mass', fix_initial=True, scaler=1.0E-5)

        phase.add_control('alt', units='km', dynamic=False, lower=0.0, upper=10.0,
                          rate_param='climb_rate', rate_continuity=True,
                          rate2_param='climb_rate2', rate2_continuity=True)

        phase.add_control('TAS', units='m/s', dynamic=False, lower=0.0, upper=260.0,
                          rate_param='TAS_rate', rate_continuity=True)

        phase.add_objective('time')

        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 100.0
        p['phase0.states:range'] = phase.interpolate(ys=(10, 35), nodes='state_disc')
        p['phase0.states:mass'] = phase.interpolate(ys=(200000, 200000), nodes='state_disc')
        p['phase0.controls:TAS'] = 250.0
        p['phase0.controls:alt'] = 9.144

        return p


if __name__ == '__main__':

    p = ex_aircraft_mission()
    p.run_driver()

    exp_out = p.model.phase0.simulate(times=np.linspace(0, 100, 100))

    import matplotlib.pyplot as plt
    plt.plot(p.model.phase0.get_values('time'), p.model.phase0.get_values('range'), 'ro')
    plt.plot(exp_out.get_values('time'), exp_out.get_values('range'), 'b-')
    plt.show()
