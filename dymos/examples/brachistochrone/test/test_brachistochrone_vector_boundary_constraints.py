import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_vector_states_ode \
    import BrachistochroneVectorStatesODE

SHOW_PLOTS = True


class TestBrachistochroneVectorBoundaryConstraints(unittest.TestCase):

    def test_brachistochrone_vector_boundary_constraints_radau_no_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos',
                        rate_source=BrachistochroneVectorStatesODE.states['pos']['rate_source'],
                        fix_initial=True, fix_final=False)
        phase.add_state('v',
                        rate_source=BrachistochroneVectorStatesODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False)

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

        p['phase0.states:pos'] = phase.interpolate(ys=[pos0, posf], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p.get_val('phase0.time')[-1], 1.8016, tolerance=1.0E-3)

        # Plot results
        if SHOW_PLOTS:
            p.run_driver()
            exp_out = phase.simulate(times_per_seg=10)

            fig, ax = plt.subplots()
            fig.suptitle('Brachistochrone Solution')

            x_imp = p.get_val('phase0.timeseries.states:pos')[:, 0]
            y_imp = p.get_val('phase0.timeseries.states:pos')[:, 1]

            x_exp = exp_out.get_val('phase0.timeseries.states:pos')[:, 0]
            y_exp = exp_out.get_val('phase0.timeseries.states:pos')[:, 1]

            ax.plot(x_imp, y_imp, 'ro', label='implicit')
            ax.plot(x_exp, y_exp, 'b-', label='explicit')

            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.grid(True)
            ax.legend(loc='upper right')

            fig, ax = plt.subplots()
            fig.suptitle('Brachistochrone Solution')

            x_imp = p.get_val('phase0.timeseries.time')
            y_imp = p.get_val('phase0.timeseries.control_rates:theta_rate2')

            x_exp = exp_out.get_val('phase0.timeseries.time')
            y_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate2')

            ax.plot(x_imp, y_imp, 'ro', label='implicit')
            ax.plot(x_exp, y_exp, 'b-', label='explicit')

            ax.set_xlabel('time (s)')
            ax.set_ylabel('theta rate2 (rad/s**2)')
            ax.grid(True)
            ax.legend(loc='lower right')

            plt.show()

        return p

    def test_brachistochrone_vector_boundary_constraints_radau_full_indices(self):

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos',
                        rate_source=BrachistochroneVectorStatesODE.states['pos']['rate_source'],
                        fix_initial=True, fix_final=False)
        phase.add_state('v',
                        rate_source=BrachistochroneVectorStatesODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False)

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

        p['phase0.states:pos'] = phase.interpolate(ys=[pos0, posf], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p.get_val('phase0.time')[-1], 1.8016, tolerance=1.0E-3)

        # Plot results
        if SHOW_PLOTS:
            p.run_driver()
            exp_out = phase.simulate(times_per_seg=10)

            fig, ax = plt.subplots()
            fig.suptitle('Brachistochrone Solution')

            x_imp = p.get_val('phase0.timeseries.states:pos')[:, 0]
            y_imp = p.get_val('phase0.timeseries.states:pos')[:, 1]

            x_exp = exp_out.get_val('phase0.timeseries.states:pos')[:, 0]
            y_exp = exp_out.get_val('phase0.timeseries.states:pos')[:, 1]

            ax.plot(x_imp, y_imp, 'ro', label='implicit')
            ax.plot(x_exp, y_exp, 'b-', label='explicit')

            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.grid(True)
            ax.legend(loc='upper right')

            fig, ax = plt.subplots()
            fig.suptitle('Brachistochrone Solution')

            x_imp = p.get_val('phase0.timeseries.time')
            y_imp = p.get_val('phase0.timeseries.control_rates:theta_rate2')

            x_exp = exp_out.get_val('phase0.timeseries.time')
            y_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate2')

            ax.plot(x_imp, y_imp, 'ro', label='implicit')
            ax.plot(x_exp, y_exp, 'b-', label='explicit')

            ax.set_xlabel('time (s)')
            ax.set_ylabel('theta rate2 (rad/s**2)')
            ax.grid(True)
            ax.legend(loc='lower right')

            plt.show()

        return p

    def test_brachistochrone_vector_boundary_constraints_radau_partial_indices(self):

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos',
                        rate_source=BrachistochroneVectorStatesODE.states['pos']['rate_source'],
                        fix_initial=True, fix_final=[True, False])
        phase.add_state('v',
                        rate_source=BrachistochroneVectorStatesODE.states['v']['rate_source'],
                        fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g',
                            units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('pos', loc='final', equals=5, indices=[1])

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        pos0 = [0, 10]
        posf = [10, 5]

        p['phase0.states:pos'] = phase.interpolate(ys=[pos0, posf], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
        p['phase0.parameters:g'] = 9.80665

        p.run_driver()

        assert_near_equal(p.get_val('phase0.time')[-1], 1.8016, tolerance=1.0E-3)

        # Plot results
        if SHOW_PLOTS:
            p.run_driver()
            exp_out = phase.simulate(times_per_seg=20)

            fig, ax = plt.subplots()
            fig.suptitle('Brachistochrone Solution')

            x_imp = p.get_val('phase0.timeseries.states:pos')[:, 0]
            y_imp = p.get_val('phase0.timeseries.states:pos')[:, 1]

            x_exp = exp_out.get_val('phase0.timeseries.states:pos')[:, 0]
            y_exp = exp_out.get_val('phase0.timeseries.states:pos')[:, 1]

            ax.plot(x_imp, y_imp, 'ro', label='implicit')
            ax.plot(x_exp, y_exp, 'b-', label='explicit')

            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.grid(True)
            ax.legend(loc='upper right')

            fig, ax = plt.subplots()
            fig.suptitle('Brachistochrone Solution')

            x_imp = p.get_val('phase0.timeseries.time')
            y_imp = p.get_val('phase0.timeseries.control_rates:theta_rate2')

            x_exp = exp_out.get_val('phase0.timeseries.time')
            y_exp = exp_out.get_val('phase0.timeseries.control_rates:theta_rate2')

            ax.plot(x_imp, y_imp, 'ro', label='implicit')
            ax.plot(x_exp, y_exp, 'b-', label='explicit')

            ax.set_xlabel('time (s)')
            ax.set_ylabel('theta rate2 (rad/s**2)')
            ax.grid(True)
            ax.legend(loc='lower right')

            plt.show()

        return p


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
