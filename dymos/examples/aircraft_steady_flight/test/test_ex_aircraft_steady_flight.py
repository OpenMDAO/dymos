import os
import unittest

import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.utils.lgl import lgl
from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE


def ex_aircraft_steady_flight(optimizer='SLSQP', solve_segments=False,
                              use_boundary_constraints=False, compressed=False):
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    _, optimizer = set_pyoptsparse_opt(optimizer, fallback=False)
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()
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

    phase = dm.Phase(ode_class=AircraftODE,
                     transcription=dm.Radau(num_segments=num_seg, segment_ends=seg_ends,
                                            order=3, compressed=compressed,
                                            solve_segments=solve_segments))

    # Pass Reference Area from an external source
    assumptions = p.model.add_subsystem('assumptions', om.IndepVarComp())
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
        phase.add_boundary_constraint('mass_fuel', loc='final',
                                      equals=1e-3, linear=False)
        phase.add_boundary_constraint('alt', loc='final', equals=10.0, linear=False)

    phase.add_state('range', units='NM',
                    rate_source='range_rate_comp.dXdt:range',
                    fix_initial=True, fix_final=False, ref=1e-3,
                    defect_ref=1e-3, lower=0, upper=2000)

    phase.add_state('mass_fuel', units='lbm',
                    rate_source='propulsion.dXdt:mass_fuel',
                    fix_initial=True, fix_final=fix_final,
                    upper=1.5E5, lower=0.0, ref=1e2, defect_ref=1e2)

    phase.add_state('alt', units='kft',
                    rate_source='climb_rate',
                    fix_initial=True, fix_final=fix_final,
                    lower=0.0, upper=60, ref=1e-3, defect_ref=1e-3)

    phase.add_control('climb_rate', units='ft/min', opt=True, lower=-3000, upper=3000,
                      targets=['gam_comp.climb_rate'],
                      rate_continuity=True, rate2_continuity=False)

    phase.add_control('mach', targets=['tas_comp.mach', 'aero.mach'], opt=False)

    phase.add_parameter('S',
                        targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'],
                        units='m**2')

    phase.add_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
    phase.add_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')

    phase.add_path_constraint('propulsion.tau', lower=0.01, upper=2.0)

    p.model.connect('assumptions.S', 'phase0.parameters:S')
    p.model.connect('assumptions.mass_empty', 'phase0.parameters:mass_empty')
    p.model.connect('assumptions.mass_payload', 'phase0.parameters:mass_payload')

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

    dm.run_problem(p)

    return p


@use_tempdirs
class TestExSteadyAircraftFlight(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'test_ex_aircraft_steady_flight_rec.db', 'SLSQP.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_ex_aircraft_steady_flight_opt(self):
        p = ex_aircraft_steady_flight(optimizer='SLSQP', solve_segments=False)
        assert_near_equal(p.get_val('phase0.timeseries.states:range', units='NM')[-1],
                          726.85, tolerance=1.0E-2)

    def test_ex_aircraft_steady_flight_solve(self):
        p = ex_aircraft_steady_flight(optimizer='SLSQP', solve_segments='forward',
                                      use_boundary_constraints=True)
        assert_near_equal(p.get_val('phase0.timeseries.states:range', units='NM')[-1],
                          726.85, tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
