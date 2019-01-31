from __future__ import print_function, division, absolute_import

import unittest

import matplotlib
matplotlib.use('Agg')

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

SHOW_PLOTS = True


class TestTimeseriesOutput(unittest.TestCase):

    def test_timeseries_gl(self):
        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_driver()

        gd = phase.grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']
        col_idxs = gd.subset_node_indices['col']

        assert_rel_error(self,
                         p.get_val('phase0.time'),
                         p.get_val('phase0.timeseries.time')[:, 0])

        assert_rel_error(self,
                         p.get_val('phase0.time_phase'),
                         p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_rel_error(self,
                             p.get_val('phase0.states:{0}'.format(state)),
                             p.get_val('phase0.timeseries.states:'
                                       '{0}'.format(state))[state_input_idxs])

            assert_rel_error(self,
                             p.get_val('phase0.state_interp.state_col:{0}'.format(state)),
                             p.get_val('phase0.timeseries.states:'
                                       '{0}'.format(state))[col_idxs])

        for control in ('theta',):
            assert_rel_error(self,
                             p.get_val('phase0.controls:{0}'.format(control)),
                             p.get_val('phase0.timeseries.controls:'
                                       '{0}'.format(control))[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                assert_rel_error(self,
                                 p.get_val('phase0.design_parameters:{0}'.format(dp))[0, :],
                                 p.get_val('phase0.timeseries.design_parameters:{0}'.format(dp))[i])

    def test_timeseries_radau(self):
        p = Problem(model=Group())

        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('radau-ps',
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_driver()

        gd = phase.grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']

        assert_rel_error(self,
                         p.get_val('phase0.time'),
                         p.get_val('phase0.timeseries.time')[:, 0])

        assert_rel_error(self,
                         p.get_val('phase0.time_phase'),
                         p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_rel_error(self,
                             p.get_val('phase0.states:{0}'.format(state)),
                             p.get_val('phase0.timeseries.states:'
                                       '{0}'.format(state))[state_input_idxs])

        for control in ('theta',):
            assert_rel_error(self,
                             p.get_val('phase0.controls:{0}'.format(control)),
                             p.get_val('phase0.timeseries.controls:'
                                       '{0}'.format(control))[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                assert_rel_error(self,
                                 p.get_val('phase0.design_parameters:{0}'.format(dp))[0, :],
                                 p.get_val('phase0.timeseries.design_parameters:'
                                           '{0}'.format(dp))[i])

    def test_timeseries_explicit(self):
        p = Problem(model=Group())
        NUM_SEG = 8
        NUM_STEPS = 10

        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('explicit',
                      ode_class=BrachistochroneODE,
                      num_segments=NUM_SEG,
                      num_steps=NUM_STEPS,
                      shooting='multiple',
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=False)
        phase.set_state_options('y', fix_initial=True, fix_final=False)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.design_parameters:g'] = 9.80665

        p.run_driver()

        gd = phase.grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']
        seg_idxs = gd.subset_segment_indices

        step_times = np.empty(0, dtype=float)
        step_time_phases = np.empty(0, dtype=float)
        for i in range(NUM_SEG):
            i1, i2 = seg_idxs['all'][i, :]

            time_segi = p.get_val('phase0.time')[i1:i2]
            step_times_segi = np.linspace(time_segi[0], time_segi[-1], NUM_STEPS+1)
            step_times = np.concatenate((step_times, step_times_segi))

            time_phase_segi = p.get_val('phase0.time_phase')[i1:i2]
            step_time_phase_segi = np.linspace(time_phase_segi[0], time_phase_segi[-1], NUM_STEPS+1)
            step_time_phases = np.concatenate((step_time_phases, step_time_phase_segi))

        assert_rel_error(self,
                         p.get_val('phase0.timeseries.time')[:, 0],
                         step_times)

        assert_rel_error(self,
                         p.get_val('phase0.timeseries.time_phase')[:, 0],
                         step_time_phases)

        # TODO: test the explicit time series using the analytic solution to the brachistochrone
        # for state in ('x', 'y', 'v'):
        #     print(state)
        #     print(p.get_val('phase0.states:{0}'.format(state)))
        #     p.get_val('phase0.timeseries.states:{0}'.format(state))[state_input_idxs]
        #     assert_rel_error(self,
        #                      p.get_val('phase0.states:{0}'.format(state)),
        #                      p.get_val('phase0.timeseries.states:'
        #                                '{0}'.format(state))[state_input_idxs])
        #
        #     step_states = np.empty(0, dtype=float)
        #     for i in range(NUM_SEG):
        #         i1, i2 = seg_idxs['all'][i, :]
        #
        #         states_segi = p.get_val('phase0.states:{0}'.format(state))[i1:i2]
        #         step_times_segi = np.linspace(time_segi[0], time_segi[-1], NUM_STEPS+1)
        #         step_times = np.concatenate((step_times, step_times_segi))
        #
        #         time_phase_segi = p.get_val('phase0.time_phase')[i1:i2]
        #         step_time_phase_segi = np.linspace(time_phase_segi[0],
        #                                            time_phase_segi[-1], NUM_STEPS+1)
        #         step_time_phases = np.concatenate((step_time_phases, step_time_phase_segi))

        # for control in ('theta',):
        #     assert_rel_error(self,
        #                      p.get_val('phase0.controls:{0}'.format(control)),
        #                      p.get_val('phase0.timeseries.controls:'
        #                                '{0}'.format(control))[control_input_idxs])
        #
        # for dp in ('g',):
        #     for i in range(gd.subset_num_nodes['all']):
        #         assert_rel_error(self,
        #                          p.get_val('phase0.design_parameters:{0}'.format(dp))[0, :],
        #                          p.get_val('phase0.timeseries.design_parameters:'
        #                                    '{0}'.format(dp))[i])

if __name__ == '__main__':
    unittest.main()
