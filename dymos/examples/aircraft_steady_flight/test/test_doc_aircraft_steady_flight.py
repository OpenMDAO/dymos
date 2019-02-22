from __future__ import print_function, absolute_import, division

import os
import unittest

import matplotlib
matplotlib.use('Agg')


class TestSteadyAircraftFlightForDocs(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['coloring.json', 'test_doc_aircraft_steady_flight_rec.db', 'SLSQP.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_steady_aircraft_for_docs(self):
        import numpy as np
        import matplotlib.pyplot as plt

        from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, IndepVarComp
        from openmdao.utils.assert_utils import assert_rel_error

        from dymos import Phase

        from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE
        from dymos.utils.lgl import lgl

        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['dynamic_simul_derivs'] = True
        # p.driver.opt_settings['MAXIT'] = 20
        num_seg = 15
        seg_ends, _ = lgl(num_seg + 1)

        phase = Phase('gauss-lobatto',
                      ode_class=AircraftODE,
                      num_segments=num_seg,
                      segment_ends=seg_ends,
                      transcription_order=3,
                      compressed=False)

        # Pass design parameters in externally from an external source
        assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
        assumptions.add_output('mach', val=0.8, units=None)
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True,
                               duration_bounds=(300, 10000),
                               duration_ref=3600)

        phase.set_state_options('range', units='NM', fix_initial=True, fix_final=False,
                                ref=1e-3, defect_ref=1e-3,)

        phase.set_state_options('mass_fuel', units='lbm', fix_initial=True, fix_final=True,
                                upper=1.5E5, lower=0.0, ref=1e2, defect_ref=1e2)

        phase.set_state_options('alt', units='kft', fix_initial=True, fix_final=True,
                                upper=60, lower=0, ref=1e-3, defect_ref=1e-3)

        phase.add_control('climb_rate', units='ft/min', opt=True, lower=-3000, upper=3000,
                          rate_continuity=True)

        phase.add_input_parameter('mach', units=None)
        phase.add_input_parameter('S', units='m**2')
        phase.add_input_parameter('mass_empty', units='kg')
        phase.add_input_parameter('mass_payload', units='kg')

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=2.0)

        p.model.connect('assumptions.mach', 'phase0.input_parameters:mach')
        p.model.connect('assumptions.S', 'phase0.input_parameters:S')
        p.model.connect('assumptions.mass_empty', 'phase0.input_parameters:mass_empty')
        p.model.connect('assumptions.mass_payload', 'phase0.input_parameters:mass_payload')

        phase.add_objective('range', loc='final', ref=-1.0e-4)

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 3600.0
        p['phase0.states:range'] = phase.interpolate(ys=(0, 800.0), nodes='state_input')
        p['phase0.states:mass_fuel'] = phase.interpolate(ys=(30000, 0), nodes='state_input')
        p['phase0.states:alt'][:] = 10.0

        p['assumptions.mach'][:] = 0.8
        p['assumptions.S'] = 427.8
        p['assumptions.mass_empty'] = 0.15E6
        p['assumptions.mass_payload'] = 84.02869 * 400

        p.run_driver()

        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 500), record=True,
                                 record_file='test_doc_aircraft_steady_flight_rec.db')

        time_imp = p.get_val('phase0.timeseries.time', units='s')
        time_exp = exp_out.get_val('phase0.timeseries.time', units='s')

        range_imp = p.get_val('phase0.timeseries.states:range', units='NM')
        range_exp = exp_out.get_val('phase0.timeseries.states:range', units='NM')

        alt_imp = p.get_val('phase0.timeseries.states:alt', units='ft')
        alt_exp = exp_out.get_val('phase0.timeseries.states:alt', units='ft')

        climb_rate_imp = p.get_val('phase0.timeseries.controls:climb_rate', units='ft/min')
        climb_rate_exp = exp_out.get_val('phase0.timeseries.controls:climb_rate', units='ft/min')

        mass_fuel_imp = p.get_val('phase0.timeseries.states:mass_fuel', units='lbm')
        mass_fuel_exp = exp_out.get_val('phase0.timeseries.states:mass_fuel', units='lbm')

        assert_rel_error(self, range_imp[-1], 726.7, tolerance=1.0E-2)

        plt.figure()

        plt.plot(time_imp, alt_imp, 'ro')
        plt.plot(time_exp, alt_exp, 'b-')

        plt.suptitle('Altitude vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (ft)')

        plt.figure()
        plt.plot(time_imp, climb_rate_imp, 'ro')
        plt.plot(time_exp, climb_rate_exp, 'b-')

        plt.suptitle('Climb Rate vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Climb Rate (ft/min)')

        plt.figure()
        plt.plot(time_imp, mass_fuel_imp, 'ro')

        plt.plot(time_exp, mass_fuel_exp, 'b-')

        plt.suptitle('Fuel Mass vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Fuel Mass (lbm)')

        plt.show()


if __name__ == '__main__':
    unittest.main()
