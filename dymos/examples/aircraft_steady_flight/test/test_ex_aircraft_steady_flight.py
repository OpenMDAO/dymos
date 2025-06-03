import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.utils.lgl import lgl
from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE


def ex_aircraft_steady_flight(transcription, optimizer='SLSQP', use_boundary_constraints=False,
                              show_output=True):
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 50
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Linesearch tolerance'] = 0.10
        if show_output:
            p.driver.opt_settings['iSumm'] = 6
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['mu_init'] = 1e-3
        p.driver.opt_settings['max_iter'] = 200
        p.driver.opt_settings['acceptable_tol'] = 1e-6
        p.driver.opt_settings['constr_viol_tol'] = 1e-6
        p.driver.opt_settings['compl_inf_tol'] = 1e-6
        p.driver.opt_settings['acceptable_iter'] = 0
        p.driver.opt_settings['tol'] = 1e-6
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        # p.driver.opt_settings['derivative_test'] = 'first-order'
        p.driver.opt_settings['print_level'] = 0
    elif optimizer == 'SLSQP':
        p.driver.opt_settings['MAXIT'] = 100

    phase = dm.Phase(ode_class=AircraftODE,
                     transcription=transcription)

    # Pass Reference Area from an external source
    assumptions = p.model.add_subsystem('assumptions', om.IndepVarComp())
    assumptions.add_output('S', val=427.8, units='m**2')
    assumptions.add_output('mass_empty', val=1.0, units='kg')
    assumptions.add_output('mass_payload', val=1.0, units='kg')

    traj = p.model.add_subsystem('traj', dm.Trajectory())
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True,
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
                    fix_initial=True, fix_final=False, ref=1,
                    defect_ref=1, lower=0, upper=2000)

    phase.add_state('mass_fuel', units='lbm',
                    rate_source='propulsion.dXdt:mass_fuel',
                    fix_initial=True, fix_final=fix_final,
                    upper=1.5E5, lower=0.0, ref=1e4, defect_ref=1e4)

    phase.add_state('alt', units='kft',
                    rate_source='climb_rate',
                    fix_initial=True, fix_final=fix_final,
                    lower=0.0, upper=60, ref=1, defect_ref=1)

    phase.add_control('climb_rate', units='ft/min', opt=True, lower=-3000, upper=3000,
                      targets=['gam_comp.climb_rate'],
                      ref0=-10_000, ref=10_000,
                      rate_continuity=True, rate2_continuity=False)

    phase.add_control('mach', targets=['tas_comp.mach', 'aero.mach'], opt=False)

    phase.add_parameter('S',
                        targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'],
                        units='m**2')

    phase.add_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
    phase.add_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')

    phase.add_path_constraint('propulsion.tau', lower=0.01, upper=2.0)

    p.model.connect('assumptions.S', 'traj.phase0.parameters:S')
    p.model.connect('assumptions.mass_empty', 'traj.phase0.parameters:mass_empty')
    p.model.connect('assumptions.mass_payload', 'traj.phase0.parameters:mass_payload')

    phase.add_objective('range', loc='final', ref=-100.0)

    p.setup()

    phase.set_time_val(initial=0.0, duration=5000.0)
    phase.set_state_val('range', (0, 724.0))
    phase.set_state_val('mass_fuel', (30000, 1e-3))
    phase.set_state_val('alt', 10.0)
    phase.set_control_val('mach', 0.8)

    p['assumptions.S'] = 427.8
    p['assumptions.mass_empty'] = 0.15E6
    p['assumptions.mass_payload'] = 84.02869 * 400

    dm.run_problem(p, simulate=False, make_plots=True)

    return p


@use_tempdirs
class TestExSteadyAircraftFlight(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_ex_aircraft_steady_flight_opt_radau(self):
        num_seg = 15
        seg_ends, _ = lgl(num_seg + 1)

        tx = dm.Radau(num_segments=num_seg, segment_ends=seg_ends, order=3, compressed=False)
        p = ex_aircraft_steady_flight(transcription=tx, optimizer='SLSQP')
        assert_near_equal(p.get_val('traj.phase0.timeseries.range', units='NM')[-1],
                          726.85, tolerance=1.0E-2)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_ex_aircraft_steady_flight_opt_birkhoff(self):

        tx = dm.Birkhoff(num_nodes=50, grid_type='cgl',
                         solve_segments=False)
        p = ex_aircraft_steady_flight(transcription=tx, optimizer='IPOPT')
        assert_near_equal(p.get_val('traj.phase0.timeseries.range', units='NM')[-1],
                          726.85, tolerance=2.0E-2)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_ex_aircraft_steady_flight_solve_radau(self):
        num_seg = 15
        seg_ends, _ = lgl(num_seg + 1)

        tx = dm.Radau(num_segments=num_seg, segment_ends=seg_ends, order=3, compressed=False,
                      solve_segments='forward')
        p = ex_aircraft_steady_flight(transcription=tx, optimizer='SLSQP',
                                      use_boundary_constraints=True)
        assert_near_equal(p.get_val('traj.phase0.timeseries.range', units='NM')[-1],
                          726.85, tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
