from __future__ import print_function, absolute_import, division

import unittest


class TestMinTimeClimbForDocs(unittest.TestCase):

    def test_min_time_climb_for_docs_gauss_lobatto(self):
        import numpy as np

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
                      num_segments=15,
                      compressed=True,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(opt_initial=False, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                                scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

        phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                                scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

        phase.set_state_options('v', fix_initial=True, lower=10.0,
                                scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

        phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                                ref=1.0, defect_scaler=1.0, units='rad')

        phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                                scaler=1.0E-3, defect_scaler=1.0E-3)

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

        p.setup(mode='fwd', check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500

        p['phase0.states:r'] = phase.interpolate(ys=[0.0, 50000.0], nodes='state_disc')
        p['phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_disc')
        p['phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_disc')
        p['phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_disc')
        p['phase0.states:m'] = phase.interpolate(ys=[19030.468, 10000.], nodes='state_disc')
        p['phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_disc')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.model.phase0.get_values('time')[-1], 321.0, tolerance=2)

        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 100))

        fig, axes = plt.subplots(2, 1, sharex=True)

        axes[0].plot(phase.get_values('time'), phase.get_values('h'), 'ro')
        axes[0].plot(exp_out.get_values('time'), exp_out.get_values('h'), 'b-')
        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('altitude (m)')

        axes[1].plot(phase.get_values('time'), phase.get_values('alpha', units='deg'), 'ro')
        axes[1].plot(exp_out.get_values('time'), exp_out.get_values('alpha', units='deg'), 'b-')
        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('alpha (deg)')

        plt.show()


if __name__ == '__main__':
    unittest.main()
