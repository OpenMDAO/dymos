import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_vector_states_ode \
    import BrachistochroneVectorStatesODE
from dymos.utils.testing_utils import assert_timeseries_near_equal


SHOW_PLOTS = True


@use_tempdirs
class TestBrachistochroneVectorBoundaryConstraints(unittest.TestCase):

    def test_brachistochrone_vector_boundary_constraints_radau_no_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=False)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g',
                            units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('pos', loc='final', equals=[10, 5])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        pos0 = [0, 10]
        posf = [10, 5]

        p['phase0.states:pos'] = phase.interp('pos', [pos0, posf])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p.get_val('phase0.t')[-1], 1.8016, tolerance=1.0E-3)

        exp_out = phase.simulate(times_per_seg=10)

        x_imp = p.get_val('phase0.timeseries.time')
        y_imp = p.get_val('phase0.control_rates:theta_rate2')

        x_exp = exp_out.get_val('phase0.timeseries.time')
        y_exp = exp_out.get_val('phase0.control_rates:theta_rate2')

        igd = phase.options['transcription'].grid_data
        egd = exp_out.model._get_subsystem('phase0').options['transcription'].options['output_grid']

        for row in range(igd.num_segments):
            istart, iend = igd.segment_indices[row, :]
            estart, eend = egd.segment_indices[row, :]
            assert_timeseries_near_equal(t_ref=x_imp[istart:iend, :],
                                         x_ref=y_imp[istart:iend, :],
                                         t_check=x_exp[estart:eend, :],
                                         x_check=y_exp[estart:eend, :],
                                         rel_tolerance=1.0E-9)

        return p

    def test_brachistochrone_vector_boundary_constraints_radau_full_indices(self):

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=False)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g',
                            units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('pos', loc='final', equals=[10, 5], indices=[0, 1])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        pos0 = [0, 10]
        posf = [10, 5]

        p['phase0.states:pos'] = phase.interp('pos', [pos0, posf])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p.get_val('phase0.t')[-1], 1.8016, tolerance=1.0E-3)

        exp_out = phase.simulate(times_per_seg=10)

        x_imp = p.get_val('phase0.timeseries.time')
        y_imp = p.get_val('phase0.control_rates:theta_rate2')

        x_exp = exp_out.get_val('phase0.timeseries.time')
        y_exp = exp_out.get_val('phase0.control_rates:theta_rate2')

        igd = phase.options['transcription'].grid_data
        egd = exp_out.model._get_subsystem('phase0').options['transcription'].options['output_grid']

        for row in range(igd.num_segments):
            istart, iend = igd.segment_indices[row, :]
            estart, eend = egd.segment_indices[row, :]
            assert_timeseries_near_equal(t_ref=x_imp[istart:iend, :],
                                         x_ref=y_imp[istart:iend, :],
                                         t_check=x_exp[estart:eend, :],
                                         x_check=y_exp[estart:eend, :],
                                         rel_tolerance=1.0E-9)

        return p

    def test_brachistochrone_vector_boundary_constraints_radau_partial_indices(self):

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=[True, False])
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('pos', loc='final', equals=5, indices=[1])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        pos0 = [0, 10]
        posf = [10, 5]

        p['phase0.states:pos'] = phase.interp('pos', [pos0, posf])
        p['phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p.get_val('phase0.t')[-1], 1.8016, tolerance=1.0E-3)

        exp_out = phase.simulate(times_per_seg=10)

        x_imp = p.get_val('phase0.timeseries.time')
        y_imp = p.get_val('phase0.control_rates:theta_rate2')

        x_exp = exp_out.get_val('phase0.timeseries.time')
        y_exp = exp_out.get_val('phase0.control_rates:theta_rate2')

        igd = phase.options['transcription'].grid_data
        egd = exp_out.model._get_subsystem('phase0').options['transcription'].options['output_grid']

        for row in range(igd.num_segments):
            istart, iend = igd.segment_indices[row, :]
            estart, eend = egd.segment_indices[row, :]
            assert_timeseries_near_equal(t_ref=x_imp[istart:iend, :],
                                         x_ref=y_imp[istart:iend, :],
                                         t_check=x_exp[estart:eend, :],
                                         x_check=y_exp[estart:eend, :],
                                         rel_tolerance=1.0E-9)

        return p


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
