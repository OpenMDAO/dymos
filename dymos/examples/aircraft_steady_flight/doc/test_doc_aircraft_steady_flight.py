from __future__ import print_function, absolute_import, division

import os
import unittest

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.style.use('ggplot')


class TestSteadyAircraftFlightForDocs(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'test_doc_aircraft_steady_flight_rec.db', 'SLSQP.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_steady_aircraft_for_docs(self):
        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_rel_error

        import dymos as dm

        from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE
        from dymos.examples.plotting import plot_results
        from dymos.utils.lgl import lgl

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        num_seg = 15
        seg_ends, _ = lgl(num_seg + 1)

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=AircraftODE,
                                        transcription=dm.Radau(num_segments=num_seg,
                                                               segment_ends=seg_ends,
                                                               order=3, compressed=False)))

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', om.IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        phase.set_time_options(initial_bounds=(0, 0),
                               duration_bounds=(300, 10000),
                               duration_ref=5600)

        phase.add_state('range', units='NM',
                        rate_source='range_rate_comp.dXdt:range',
                        fix_initial=True, fix_final=False, ref=1e-3,
                        defect_ref=1e-3, lower=0, upper=2000)

        phase.add_state('mass_fuel', units='lbm',
                        rate_source='propulsion.dXdt:mass_fuel',
                        targets=['mass_comp.mass_fuel'],
                        fix_initial=True, fix_final=True,
                        upper=1.5E5, lower=0.0, ref=1e2, defect_ref=1e2)

        phase.add_state('alt', units='kft',
                        rate_source='climb_rate',
                        targets=['atmos.h', 'aero.alt', 'propulsion.alt'],
                        fix_initial=True, fix_final=True,
                        lower=0.0, upper=60, ref=1e-3, defect_ref=1e-3)

        phase.add_control('climb_rate', units='ft/min', opt=True, lower=-3000, upper=3000,
                          targets=['gam_comp.climb_rate'],
                          rate_continuity=True, rate2_continuity=False)

        phase.add_control('mach', targets=['tas_comp.mach', 'aero.mach'], units=None, opt=False)

        phase.add_input_parameter('S',
                                  targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'],
                                  units='m**2')

        phase.add_input_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
        phase.add_input_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=2.0, shape=(1,))

        p.model.connect('assumptions.S', 'traj.phase0.input_parameters:S')
        p.model.connect('assumptions.mass_empty', 'traj.phase0.input_parameters:mass_empty')
        p.model.connect('assumptions.mass_payload', 'traj.phase0.input_parameters:mass_payload')

        phase.add_objective('range', loc='final', ref=-1.0e-4)

        p.setup()

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 3600.0
        p['traj.phase0.states:range'][:] = phase.interpolate(ys=(0, 724.0), nodes='state_input')
        p['traj.phase0.states:mass_fuel'][:] = phase.interpolate(ys=(30000, 1e-3), nodes='state_input')
        p['traj.phase0.states:alt'][:] = 10.0

        p['traj.phase0.controls:mach'][:] = 0.8

        p['assumptions.S'] = 427.8
        p['assumptions.mass_empty'] = 0.15E6
        p['assumptions.mass_payload'] = 84.02869 * 400

        p.run_driver()

        assert_rel_error(self, p.get_val('traj.phase0.timeseries.states:range', units='NM')[-1],
                         726.85, tolerance=1.0E-2)

        exp_out = traj.simulate()

        plot_results([('traj.phase0.timeseries.states:range', 'traj.phase0.timeseries.states:alt',
                       'range (NM)', 'altitude (kft)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:mass_fuel',
                       'time (s)', 'fuel mass (lbm)')],
                     title='Commercial Aircraft Optimization',
                     p_sol=p, p_sim=exp_out)

        plt.show()


if __name__ == '__main__':
    unittest.main()
