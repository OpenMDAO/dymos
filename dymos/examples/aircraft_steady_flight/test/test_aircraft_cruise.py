from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, \
    pyOptSparseDriver, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase
from dymos.examples.aircraft_steady_flight.aircraft_ode import AircraftODE

optimizer = 'SLSQP'

try:
    import MBI
except:
    MBI = None


class TestAircraftCruise(unittest.TestCase):

    def test_cruise_results(self):
        p = Problem(model=Group())
        if optimizer == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = optimizer
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major step limit'] = 0.05
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            p.driver.opt_settings["Linesearch tolerance"] = 0.10
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        else:
            p.driver = ScipyOptimizeDriver()
            p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=AircraftODE,
                      num_segments=1,
                      transcription_order=13)

        # Pass Reference Area from an external source
        assumptions = p.model.add_subsystem('assumptions', IndepVarComp())
        assumptions.add_output('S', val=427.8, units='m**2')
        assumptions.add_output('mass_empty', val=1.0, units='kg')
        assumptions.add_output('mass_payload', val=1.0, units='kg')

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0),
                               duration_bounds=(3600, 3600),
                               duration_ref=3600)

        phase.set_state_options('range', units='km', fix_initial=True, fix_final=False, scaler=0.01,
                                defect_scaler=1.0E-1)
        phase.set_state_options('mass_fuel', fix_final=True, upper=20000.0, lower=0.0,
                                scaler=1.0E-4, defect_scaler=1.0E-2)

        phase.add_control('alt', units='km', opt=False, rate_param='climb_rate')

        phase.add_control('mach', units=None, opt=False)

        phase.add_design_parameter('S', units='m**2', opt=False)
        phase.add_design_parameter('mass_empty', units='kg', opt=False)
        phase.add_design_parameter('mass_payload', units='kg', opt=False)

        phase.add_path_constraint('propulsion.tau', lower=0.01, upper=1.0)

        p.model.connect('assumptions.S', 'phase0.design_parameters:S')
        p.model.connect('assumptions.mass_empty', 'phase0.design_parameters:mass_empty')
        p.model.connect('assumptions.mass_payload', 'phase0.design_parameters:mass_payload')

        phase.add_objective('time', loc='final', ref=3600)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(mode='fwd')

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.515132 * 3600.0
        p['phase0.states:range'] = phase.interpolate(ys=(0, 1296.4), nodes='state_disc')
        p['phase0.states:mass_fuel'] = phase.interpolate(ys=(12236.594555, 0), nodes='state_disc')
        p['phase0.controls:mach'] = 0.8
        p['phase0.controls:alt'] = 5.0

        p['assumptions.S'] = 427.8
        p['assumptions.mass_empty'] = 0.15E6
        p['assumptions.mass_payload'] = 84.02869 * 400

        p.run_driver()

        tas = phase.get_values('tas_comp.TAS', units='m/s')
        time = phase.get_values('time', units='s')
        range = phase.get_values('range', units='m')

        assert_rel_error(self, range, tas*time, tolerance=1.0E-9)
