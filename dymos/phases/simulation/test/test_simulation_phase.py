from __future__ import print_function, absolute_import, division

import unittest

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error
from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.phases.simulation.simulation_phase import SimulationPhase

p = Problem(model=Group())
p.driver = ScipyOptimizeDriver()

phase = Phase('gauss-lobatto',
              ode_class=BrachistochroneODE,
              num_segments=10)

p.model.add_subsystem('phase0', phase)

phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

phase.set_state_options('x', fix_initial=True, fix_final=True)
phase.set_state_options('y', fix_initial=True, fix_final=True)
phase.set_state_options('v', fix_initial=True)

phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)

phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

# Minimize time at the end of the phase
phase.add_objective('time', loc='final', scaler=10)

phase.add_path_constraint('theta_rate', lower=0, upper=100, units='deg/s')

p.model.linear_solver = DirectSolver(assemble_jac=True)
p.model.options['assembled_jac_type'] = 'csc'

p.setup()

p['phase0.t_initial'] = 0.0
p['phase0.t_duration'] = 2.0

p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

# Solve for the optimal trajectory
p.run_driver()


class TestSimulationPhase(unittest.TestCase):

    def test_simulate_phase_gl(self):

        sim_prob = Problem(model=Group())

        sim_phase = SimulationPhase(grid_data=phase.grid_data,
                                    time_options=phase.time_options,
                                    state_options=phase.state_options,
                                    control_options=phase.control_options,
                                    design_parameter_options=phase.design_parameter_options,
                                    input_parameter_options=phase.input_parameter_options,
                                    ode_class=phase.options['ode_class'],
                                    ode_init_kwargs=phase.options['ode_init_kwargs'])

        sim_prob.model.add_subsystem(phase.name, sim_phase)
        sim_prob.setup(check=True)

        sim_prob['phase0.time'] = p.get_val('phase0.time')
        sim_prob['phase0.implicit_states:x'] = p.get_val('phase0.timeseries.states:x')
        sim_prob['phase0.implicit_states:y'] = p.get_val('phase0.timeseries.states:y')
        sim_prob['phase0.implicit_states:v'] = p.get_val('phase0.timeseries.states:v')
        sim_prob['phase0.implicit_controls:theta'] = p.get_val('phase0.timeseries.controls:theta')
        sim_prob['phase0.design_parameters:g'] = p.get_val('phase0.design_parameters:g')

        sim_prob.run_model()

        sim_prob.model.list_outputs(values=True, print_arrays=True)


if __name__ == '__main__':
    unittest.main()
