from __future__ import print_function, absolute_import, division

import unittest


class TestMinTimeClimbForDocs(unittest.TestCase):

    def test_min_time_climb_for_docs_gauss_lobatto(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error

        from dymos import Phase
        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        # Compute sparsity/coloring when run_driver is called
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=MinTimeClimbODE,
                      num_segments=12,
                      compressed=True,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                                ref=1.0E3, defect_ref=1000.0, units='m')

        phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                                ref=1.0E2, defect_ref=100.0, units='m')

        phase.set_state_options('v', fix_initial=True, lower=10.0,
                                ref=1.0E2, defect_ref=0.1, units='m/s')

        phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                                ref=1.0, defect_scaler=1.0, units='rad')

        phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                                ref=1.0E3, defect_ref=0.1)

        rate_continuity = True

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=rate_continuity, rate_continuity_scaler=100.0,
                          rate2_continuity=False)

        phase.add_design_parameter('S', val=49.2386, units='m**2', opt=False)
        phase.add_design_parameter('Isp', val=1600.0, units='s', opt=False)
        phase.add_design_parameter('throttle', val=1.0, opt=False)

        phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
        phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

        phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
        phase.add_path_constraint(name='alpha', lower=-8, upper=8)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final')

        p.driver.options['dynamic_simul_derivs'] = True
        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500

        p['phase0.states:r'] = phase.interpolate(ys=[0.0, 50000.0], nodes='state_input')
        p['phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
        p['phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
        p['phase0.states:m'] = phase.interpolate(ys=[19030.468, 10000.], nodes='state_input')
        p['phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.t_duration'), 321.0, tolerance=2)

        exp_out = phase.simulate(times=50)

        fig, axes = plt.subplots(2, 1, sharex=True)

        t_sol = p.get_val('phase0.timeseries.time')
        t_exp = exp_out.get_val('phase0.timeseries.time')

        h_sol = p.get_val('phase0.timeseries.states:h')
        h_exp = exp_out.get_val('phase0.timeseries.states:h')

        alpha_sol = p.get_val('phase0.timeseries.controls:alpha', units='deg')
        alpha_exp = exp_out.get_val('phase0.timeseries.controls:alpha', units='deg')

        axes[0].plot(t_sol, h_sol, 'ro')
        axes[0].plot(t_exp, h_exp, 'b-')
        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('altitude (m)')

        axes[1].plot(t_sol, alpha_sol, 'ro')
        axes[1].plot(t_exp, alpha_exp, 'b-')
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('alpha (deg)')

        plt.show()


if __name__ == '__main__':
    unittest.main()
