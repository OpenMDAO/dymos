import unittest

from scipy.interpolate import interp1d

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestTimeseriesOutput(unittest.TestCase):

    def test_timeseries_gl(self, test_smaller_timeseries=False):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        if test_smaller_timeseries:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665, include_timeseries=False)
        else:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        gd = phase.options['transcription'].grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']
        col_idxs = gd.subset_node_indices['col']

        assert_near_equal(p.get_val('phase0.time'),
                          p.get_val('phase0.timeseries.time')[:, 0])

        assert_near_equal(p.get_val('phase0.time_phase'),
                          p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_near_equal(p.get_val('phase0.states:{0}'.format(state)),
                              p.get_val('phase0.timeseries.states:'
                                        '{0}'.format(state))[state_input_idxs])

            assert_near_equal(p.get_val('phase0.state_interp.state_col:{0}'.format(state)),
                              p.get_val('phase0.timeseries.states:'
                                        '{0}'.format(state))[col_idxs])

        for control in ('theta',):
            assert_near_equal(p.get_val('phase0.controls:{0}'.format(control)),
                              p.get_val('phase0.timeseries.controls:'
                                        '{0}'.format(control))[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                if test_smaller_timeseries:
                    with self.assertRaises(KeyError):
                        p.get_val('phase0.timeseries.parameters:{0}'.format(dp))
                else:
                    assert_near_equal(p.get_val('phase0.parameters:{0}'.format(dp))[0],
                                      p.get_val('phase0.timeseries.parameters:{0}'.format(dp))[i])

        # call simulate to test SolveIVP transcription
        exp_out = phase.simulate()
        if test_smaller_timeseries:
            with self.assertRaises(KeyError):
                exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))
        else:  # no error accessing timseries.parameter
            exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))

    def test_timeseries_gl_smaller_timeseries(self):
        self.test_timeseries_gl(test_smaller_timeseries=True)

    def test_timeseries_radau(self, test_smaller_timeseries=False):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=8, order=3, compressed=True))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)

        phase.add_state('y', fix_initial=True, fix_final=True)

        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True, opt=True,
                          units='deg', lower=0.01, upper=179.9, ref=1, ref0=0)

        if test_smaller_timeseries:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665, include_timeseries=False)
        else:
            phase.add_parameter('g', opt=True, units='m/s**2', val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        gd = phase.options['transcription'].grid_data
        state_input_idxs = gd.subset_node_indices['state_input']
        control_input_idxs = gd.subset_node_indices['control_input']

        assert_near_equal(p.get_val('phase0.time'),
                          p.get_val('phase0.timeseries.time')[:, 0])

        assert_near_equal(p.get_val('phase0.time_phase'),
                          p.get_val('phase0.timeseries.time_phase')[:, 0])

        for state in ('x', 'y', 'v'):
            assert_near_equal(p.get_val('phase0.states:{0}'.format(state)),
                              p.get_val('phase0.timeseries.states:'
                                        '{0}'.format(state))[state_input_idxs])

        for control in ('theta',):
            assert_near_equal(p.get_val('phase0.controls:{0}'.format(control)),
                              p.get_val('phase0.timeseries.controls:'
                                        '{0}'.format(control))[control_input_idxs])

        for dp in ('g',):
            for i in range(gd.subset_num_nodes['all']):
                if test_smaller_timeseries:
                    with self.assertRaises(KeyError):
                        p.get_val('phase0.timeseries.parameters:{0}'.format(dp))
                else:
                    assert_near_equal(p.get_val('phase0.parameters:{0}'.format(dp))[0],
                                      p.get_val('phase0.timeseries.parameters:'
                                                '{0}'.format(dp))[i])

        # call simulate to test SolveIVP transcription
        exp_out = phase.simulate()
        if test_smaller_timeseries:
            with self.assertRaises(KeyError):
                exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))
        else:  # no error accessing timseries.parameter
            exp_out.get_val('phase0.timeseries.parameters:{0}'.format(dp))

        # Test that the state rates are output in both the radau and solveivp timeseries outputs
        t_sol = p.get_val('phase0.timeseries.time')
        t_sim = exp_out.get_val('phase0.timeseries.time')

        for state_name in ('x', 'y', 'v'):
            rate_sol = p.get_val(f'phase0.timeseries.state_rates:{state_name}')
            rate_sim = exp_out.get_val(f'phase0.timeseries.state_rates:{state_name}')

            rate_t_sim = interp1d(t_sim.ravel(), rate_sim.ravel())

            assert_near_equal(rate_t_sim(t_sol), rate_sol, tolerance=1.0E-3)

    def test_timeseries_radau_smaller_timeseries(self):
        self.test_timeseries_radau(test_smaller_timeseries=True)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
