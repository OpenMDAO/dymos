from __future__ import print_function, absolute_import, division

import unittest

import numpy as np

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, CaseReader
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

phase.add_timeseries_output('check', units='m/s', shape=(1,))

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

    def test_simulate_phase2(self):
        import matplotlib.pyplot as plt

        sim_out = phase.simulate(times=np.linspace(0, 1.8, 50))

        cr = CaseReader('phase0_sim.sql')
        case = cr.get_case(cr.list_cases()[0])

        plt.figure()

        plt.plot(p.get_val('phase0.timeseries.time', units='s'),
                 p.get_val('phase0.timeseries.controls:theta'),
                 'ro')

        plt.plot(sim_out.get_val('phase0.timeseries.time', units='s'),
                 sim_out.get_val('phase0.timeseries.controls:theta'),
                 'b-')

        plt.plot(case.outputs['phase0.timeseries.time'],
                 case.outputs['phase0.timeseries.controls:theta'],
                 'k+')

        plt.figure()

        plt.plot(p.get_val('phase0.timeseries.time', units='s'),
                 p.get_val('phase0.timeseries.check'),
                 'ro')

        plt.plot(sim_out.get_val('phase0.timeseries.time', units='s'),
                 sim_out.get_val('phase0.timeseries.check'),
                 'b-')

        plt.figure()

        plt.plot(p.get_val('phase0.timeseries.states:x', units='m'),
                 p.get_val('phase0.timeseries.states:y', units='m'),
                 'ro')

        plt.plot(sim_out.get_val('phase0.timeseries.states:x', units='m'),
                 sim_out.get_val('phase0.timeseries.states:y', units='m'),
                 'b-')

        plt.show()

    def test_simulate_phase_gl(self):
        sim_prob = Problem(model=Group())

        sim_phase = SimulationPhase(grid_data=phase.grid_data,
                                    ode_class=phase.options['ode_class'],
                                    ode_init_kwargs=phase.options['ode_init_kwargs'],
                                    times=np.linspace(0, 1.8, 1000),
                                    t_initial=p.get_val('phase0.t_initial'),
                                    t_duration=p.get_val('phase0.t_duration'),
                                    timeseries_outputs=phase._timeseries_outputs)

        sim_phase.time_options.update(phase.time_options)
        sim_phase.state_options.update(phase.state_options)
        sim_phase.control_options.update(phase.control_options)
        sim_phase.design_parameter_options.update(phase.design_parameter_options)
        sim_phase.input_parameter_options.update(phase.input_parameter_options)

        sim_prob.model.add_subsystem(phase.name, sim_phase)
        sim_prob.setup(check=True)

        # sim_prob['phase0.time'] = p.get_val('phase0.time')
        sim_prob['phase0.initial_states:x'] = p.get_val('phase0.timeseries.states:x')[0, ...]
        sim_prob['phase0.initial_states:y'] = p.get_val('phase0.timeseries.states:y')[0, ...]
        sim_prob['phase0.initial_states:v'] = p.get_val('phase0.timeseries.states:v')[0, ...]
        sim_prob['phase0.implicit_controls:theta'] = p.get_val('phase0.timeseries.controls:theta')
        sim_prob['phase0.design_parameters:g'] = p.get_val('phase0.design_parameters:g')

        sim_prob.run_model()

        # TODO: assert results with interpolation here

        # import matplotlib.pyplot as plt
        #
        # plt.figure()
        #
        # plt.plot(p.get_val('phase0.timeseries.time', units='s'),
        #          p.get_val('phase0.timeseries.controls:theta'),
        #          'ro')
        #
        # plt.plot(sim_prob.get_val('phase0.timeseries.time', units='s'),
        #          sim_prob.get_val('phase0.timeseries.controls:theta'),
        #          'b-')
        #
        # plt.figure()
        #
        # plt.plot(p.get_val('phase0.timeseries.time', units='s'),
        #          p.get_val('phase0.timeseries.check'),
        #          'ro')
        #
        # plt.plot(sim_prob.get_val('phase0.timeseries.time', units='s'),
        #          sim_prob.get_val('phase0.timeseries.check'),
        #          'b-')
        #
        # plt.figure()
        #
        # plt.plot(p.get_val('phase0.timeseries.states:x', units='m'),
        #          p.get_val('phase0.timeseries.states:y', units='m'),
        #          'ro')
        #
        # plt.plot(sim_prob.get_val('phase0.timeseries.states:x', units='m'),
        #          sim_prob.get_val('phase0.timeseries.states:y', units='m'),
        #          'b-')
        #
        # plt.show()


if __name__ == '__main__':
    unittest.main()
