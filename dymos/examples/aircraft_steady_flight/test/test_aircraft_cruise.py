import os
import pathlib
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE

optimizer = os.environ.get('DYMOS_DEFAULT_OPT', 'SLSQP')


@use_tempdirs
class TestAircraftCruise(unittest.TestCase):

    def test_cruise_results_gl(self):
        p = om.Problem()
        if optimizer == 'SNOPT':
            p.driver = om.pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.declare_coloring()
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major step limit'] = 0.05
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            p.driver.opt_settings["Linesearch tolerance"] = 0.10
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = om.ScipyOptimizeDriver()
            p.driver.declare_coloring()

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', om.IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        transcription = dm.GaussLobatto(num_segments=1,
                                        order=13,
                                        compressed=False)
        phase = dm.Phase(ode_class=AircraftODE, transcription=transcription)
        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True,
                               duration_bounds=(3600, 3600),
                               duration_ref=3600)

        phase.add_state('range', units='km', fix_initial=True, fix_final=False, scaler=0.01,
                        rate_source='range_rate_comp.dXdt:range',
                        defect_scaler=0.01)
        phase.add_state('mass_fuel', units='kg', fix_final=True, upper=20000.0, lower=0.0,
                        rate_source='propulsion.dXdt:mass_fuel',
                        scaler=1.0E-4, defect_scaler=1.0E-2)
        phase.add_state('alt',
                        rate_source='climb_rate',
                        units='km', fix_initial=True)

        phase.add_control('mach', targets=['tas_comp.mach', 'aero.mach'], units=None, opt=False)

        phase.add_control('climb_rate', targets=['gam_comp.climb_rate'], units='m/s', opt=False)

        phase.add_parameter('S',
                            targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'],
                            units='m**2')

        phase.add_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
        phase.add_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)

        phase.add_timeseries_output('tas_comp.TAS')

        p.model.connect('assumptions.S', 'phase0.parameters:S')
        p.model.connect('assumptions.mass_empty', 'phase0.parameters:mass_empty')
        p.model.connect('assumptions.mass_payload', 'phase0.parameters:mass_payload')

        phase.add_objective('time', loc='final', ref=3600)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=3600)
        phase.set_state_val('range', (0, 1296.4))
        phase.set_state_val('mass_fuel', (12236.594555, 1))
        phase.set_state_val('alt', 5.0)
        phase.set_control_val('mach', 0.8)
        phase.set_control_val('climb_rate', 0.0)

        p['assumptions.S'] = 427.8
        p['assumptions.mass_empty'] = 0.15E6
        p['assumptions.mass_payload'] = 84.02869 * 400

        dm.run_problem(p)

        time = p.get_val('phase0.timeseries.time')
        tas = p.get_val('phase0.timeseries.TAS', units='km/s')
        range = p.get_val('phase0.timeseries.range')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)

        exp_out = phase.simulate()

        time = exp_out.get_val('phase0.timeseries.time')
        tas = exp_out.get_val('phase0.timeseries.TAS', units='km/s')
        range = exp_out.get_val('phase0.timeseries.range')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)

    def test_cruise_results_radau(self):
        p = om.Problem(model=om.Group())
        if optimizer == 'SNOPT':
            p.driver = om.pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.declare_coloring()
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major step limit'] = 0.05
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            p.driver.opt_settings["Linesearch tolerance"] = 0.10
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = om.ScipyOptimizeDriver()
            p.driver.declare_coloring()

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', om.IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        transcription = dm.Radau(num_segments=1, order=13, compressed=False)
        phase = dm.Phase(ode_class=AircraftODE, transcription=transcription)
        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True,
                               duration_bounds=(3600, 3600),
                               duration_ref=3600)

        phase.add_state('range', units='km', fix_initial=True, fix_final=False, scaler=0.01,
                        rate_source='range_rate_comp.dXdt:range',
                        defect_scaler=0.01)
        phase.add_state('mass_fuel', units='kg', fix_final=True, upper=20000.0, lower=0.0,
                        rate_source='propulsion.dXdt:mass_fuel',
                        scaler=1.0E-4, defect_scaler=1.0E-2)
        phase.add_state('alt',
                        rate_source='climb_rate',
                        units='km', fix_initial=True)

        phase.add_control('mach', targets=['tas_comp.mach', 'aero.mach'], opt=False)

        phase.add_control('climb_rate', targets=['gam_comp.climb_rate'], units='m/s', opt=False)

        phase.add_parameter('S',
                            targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'],
                            units='m**2')

        phase.add_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
        phase.add_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)

        phase.add_timeseries_output('tas_comp.TAS')

        p.model.connect('assumptions.S', 'phase0.parameters:S')
        p.model.connect('assumptions.mass_empty', 'phase0.parameters:mass_empty')
        p.model.connect('assumptions.mass_payload', 'phase0.parameters:mass_payload')

        phase.add_objective('time', loc='final', ref=3600)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=3600)
        phase.set_state_val('range', (0, 1296.4))
        phase.set_state_val('mass_fuel', (12236.594555, 0))
        phase.set_state_val('alt', 5.0)
        phase.set_control_val('mach', 0.8)
        phase.set_control_val('climb_rate', 0.0)

        p['assumptions.S'] = 427.8
        p['assumptions.mass_empty'] = 0.15E6
        p['assumptions.mass_payload'] = 84.02869 * 400

        dm.run_problem(p)

        time = p.get_val('phase0.timeseries.time')
        tas = p.get_val('phase0.timeseries.TAS', units='km/s')
        range = p.get_val('phase0.timeseries.range')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)

        exp_out = phase.simulate()

        time = exp_out.get_val('phase0.timeseries.time')
        tas = exp_out.get_val('phase0.timeseries.TAS', units='km/s')
        range = exp_out.get_val('phase0.timeseries.range')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)

    def test_cruise_results_birkhoff(self):
        p = om.Problem(model=om.Group())
        optimizer = 'SLSQP'
        if optimizer == 'SNOPT':
            p.driver = om.pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.declare_coloring()
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major step limit'] = 0.05
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            p.driver.opt_settings["Linesearch tolerance"] = 0.10
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = om.ScipyOptimizeDriver()
            p.driver.declare_coloring()

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', om.IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        transcription = dm.Birkhoff(num_nodes=50, grid_type='lgl', solve_segments='forward')
        phase = dm.Phase(ode_class=AircraftODE, transcription=transcription)
        traj = dm.Trajectory()
        traj.add_phase('phase0', phase)
        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True,
                               duration_bounds=(3600, 3600),
                               duration_ref=3600)

        phase.add_state('range', units='km', fix_initial=True, fix_final=False, scaler=0.01,
                        rate_source='range_rate_comp.dXdt:range',
                        defect_scaler=0.01)
        phase.add_state('mass_fuel', units='kg', fix_final=False, upper=20000.0, lower=0.0,
                        rate_source='propulsion.dXdt:mass_fuel',
                        scaler=1.0E-4, defect_scaler=1.0E-2)
        phase.add_state('alt',
                        rate_source='climb_rate',
                        units='km', fix_initial=True)

        phase.add_control('mach', targets=['tas_comp.mach', 'aero.mach'], opt=False)

        phase.add_control('climb_rate', targets=['gam_comp.climb_rate'], units='m/s', opt=False)

        phase.add_parameter('S',
                            targets=['aero.S', 'flight_equilibrium.S', 'propulsion.S'],
                            units='m**2')

        phase.add_parameter('mass_empty', targets=['mass_comp.mass_empty'], units='kg')
        phase.add_parameter('mass_payload', targets=['mass_comp.mass_payload'], units='kg')

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)
        phase.add_boundary_constraint('mass_fuel', loc='final', equals=0.)
        phase.add_boundary_constraint('range_rate_comp.dXdt:range', loc='final', lower=0.)
        phase.add_boundary_constraint('range_rate_comp.dXdt:range', loc='initial', lower=0.)

        phase.add_timeseries_output('tas_comp.TAS')

        p.model.connect('assumptions.S', 'traj.phase0.parameters:S')
        p.model.connect('assumptions.mass_empty', 'traj.phase0.parameters:mass_empty')
        p.model.connect('assumptions.mass_payload', 'traj.phase0.parameters:mass_payload')

        phase.add_objective('time', loc='final', ref=3600)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=3600.0)
        phase.set_state_val('range', [0.0, 500.0])
        phase.set_state_val('mass_fuel', [10000.0, 1.0])
        phase.set_state_val('alt', [10.0, 10.0])
        phase.set_control_val('mach', 0.8)
        phase.set_control_val('climb_rate', 0.0)

        p.set_val('assumptions.S', 427.8)
        p.set_val('assumptions.mass_empty', 0.15E6)
        p.set_val('assumptions.mass_payload', 84.02869 * 400)

        dm.run_problem(p, simulate=True, make_plots=True)

        time = p.get_val('traj.phase0.timeseries.time')
        tas = p.get_val('traj.phase0.timeseries.TAS', units='km/s')
        range = p.get_val('traj.phase0.timeseries.range')
        mass_fuel = p.get_val('traj.phase0.timeseries.mass_fuel')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)
        assert_near_equal(mass_fuel[-1, ...], 0.0, tolerance=1.0E-4)

        sim_out_dir = traj.sim_prob.get_outputs_dir()
        case = om.CaseReader(sim_out_dir / 'dymos_simulation.db').get_case('final')

        time = case.get_val('traj.phase0.timeseries.time')
        tas = case.get_val('traj.phase0.timeseries.TAS', units='km/s')
        range = case.get_val('traj.phase0.timeseries.range')
        mass_fuel = case.get_val('traj.phase0.timeseries.mass_fuel')

        assert_near_equal(range, tas*time, tolerance=1.0E-4)
        assert_near_equal(mass_fuel[-1, ...], 0.0, tolerance=1.0E-4)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
