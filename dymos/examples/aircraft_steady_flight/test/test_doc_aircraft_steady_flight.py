from __future__ import print_function, absolute_import, division

import os
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class TestSteadyAircraftFlightForDocs(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['coloring.json', 'test_doc_aircraft_steady_flight_rec.db', 'SLSQP.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_steady_aircraft_for_docs(self):
        import matplotlib.pyplot as plt

        from openmdao.api import Problem, Group, pyOptSparseDriver, IndepVarComp
        from openmdao.utils.assert_utils import assert_rel_error

        from dymos import Phase, Radau

        from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE
        from dymos.utils.lgl import lgl

        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['dynamic_simul_derivs'] = True

        num_seg = 15
        seg_ends, _ = lgl(num_seg + 1)

        phase = Phase(ode_class=AircraftODE,
                      transcription=Radau(num_segments=num_seg, segment_ends=seg_ends,
                                          order=3, compressed=False))

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
                                defect_ref=1e-3, lower=0, upper=2000)
        phase.set_state_options('mass_fuel', units='lbm', fix_initial=True, fix_final=True,
                                upper=1.5E5, lower=0.0, ref=1e2, defect_ref=1e2)
        phase.set_state_options('alt', units='kft', fix_initial=True, fix_final=True,
                                lower=0.0, upper=60, ref=1e-3, defect_ref=1e-3)

        phase.add_control('climb_rate', units='ft/min', opt=True, lower=-3000, upper=3000,
                          rate_continuity=True, rate2_continuity=False)

        phase.add_control('mach', units=None, opt=False)

        phase.add_input_parameter('S', units='m**2')
        phase.add_input_parameter('mass_empty', units='kg')
        phase.add_input_parameter('mass_payload', units='kg')

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=2.0, shape=(1,))

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

        assert_rel_error(self, p.get_val('phase0.timeseries.states:range', units='NM')[-1],
                         726.85, tolerance=1.0E-2)

        exp_out = phase.simulate()

        t_imp = p.get_val('phase0.timeseries.time')
        t_exp = exp_out.get_val('phase0.timeseries.time')

        alt_imp = p.get_val('phase0.timeseries.states:alt')
        alt_exp = exp_out.get_val('phase0.timeseries.states:alt')

        climb_rate_imp = p.get_val('phase0.timeseries.controls:climb_rate', units='ft/min')
        climb_rate_exp = exp_out.get_val('phase0.timeseries.controls:climb_rate',
                                         units='ft/min')

        mass_fuel_imp = p.get_val('phase0.timeseries.states:mass_fuel', units='kg')
        mass_fuel_exp = exp_out.get_val('phase0.timeseries.states:mass_fuel', units='kg')

        plt.show()
        plt.plot(t_imp, alt_imp, 'b-')
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


if __name__ == '__main__':
    unittest.main()
