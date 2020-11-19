import unittest

import matplotlib
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs


class TestBalancedFieldLengthForDocs(unittest.TestCase):

    # @save_for_docs
    def test_balanced_field_length_for_docs(self):
        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.balanced_field.ground_roll_ode import GroundRollODE

        #
        # Instantiate the problem and configure the optimization driver
        #
        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['Major iterations limit'] = 20
        p.driver.declare_coloring()

        #
        # Instantiate the trajectory and phase
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', traj)

        #
        # First Phase - Brake Release to V1
        # We don't know what V1 is a priori, we're going to use this model to determine it.
        #

        p1 = dm.Phase(ode_class=GroundRollODE, transcription=dm.Radau(num_segments=10))

        traj.add_phase('brake_release_to_v1', p1)

        #
        # Set the options on the optimization variables
        #
        p1.set_time_options(fix_initial=True, duration_bounds=(40, 100), duration_ref=10.0)

        p1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0,
                     rate_source='r_dot')

        p1.add_state('v', fix_initial=True, lower=0.0, ref=100.0, defect_ref=100.0, rate_source='v_dot')

        p1.add_parameter('h', opt=False, units='m')
        p1.add_parameter('T', val=120101.98, opt=False, units='N')
        p1.add_parameter('alpha', val=0.0, opt=False, units='deg')

        p1.add_timeseries_output('*')
        
        # Second Phase - V1 to Vr
        # Vr is taken to be 1.2 * the stall speed (v_stall)
        #

        p2 = dm.Phase(ode_class=GroundRollODE,
                         transcription=dm.Radau(num_segments=10))

        traj.add_phase('v1_to_vr', p2)

        #
        # Set the options on the optimization variables
        #
        p2.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)

        p2.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
                     rate_source='r_dot')

        p2.add_state('v', fix_initial=False, lower=0.0, ref=100.0, defect_ref=100.0,
                     rate_source='v_dot')

        p2.add_parameter('h', val=0.0, opt=False, units='m')
        p2.add_parameter('T', val=120101.98/2, opt=False, units='N')
        p2.add_parameter('alpha', val=0.0, opt=False, units='deg')

        p2.add_timeseries_output('*')

        p2.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2)

        # Minimize time at the end of the phase
        # p2.add_objective('time', loc='final', ref=1.0)
        
        # Third Phase - Rejected Takeoff
        # V1 to Zero speed with no propulsion and braking.
        #

        p3 = dm.Phase(ode_class=GroundRollODE, transcription=dm.Radau(num_segments=10))

        traj.add_phase('rto', p3)

        #
        # Set the options on the optimization variables
        #
        p3.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)

        p3.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
                     rate_source='r_dot')

        p3.add_state('v', fix_initial=False, lower=0.0, ref=100.0, defect_ref=100.0,
                     rate_source='v_dot')

        p3.add_parameter('h', val=0.0, opt=False, units='m')
        p3.add_parameter('T', val=0.0, opt=False, units='N')
        p3.add_parameter('mu_r', val=0.3, opt=False, units=None)
        p3.add_parameter('alpha', val=0.0, opt=False, units='deg')

        p3.add_timeseries_output('*')

        p3.add_boundary_constraint('v', loc='final', equals=0, ref=100)

        # Minimize range at the end of the phase
        p3.add_objective('time', loc='final', ref=100.0)
        

        # Third Phase - Rejected Takeoff
        # V1 to Zero speed with no propulsion and braking.
        #

        p4 = dm.Phase(ode_class=GroundRollODE, transcription=dm.Radau(num_segments=10))

        traj.add_phase('rotate', p4)

        #
        # Set the options on the optimization variables
        #
        p4.set_time_options(fix_initial=False, duration_bounds=(0.1, 20), duration_ref=1.0)

        p4.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
                     rate_source='r_dot')

        p4.add_state('v', fix_initial=False, lower=0.0, ref=100.0, defect_ref=100.0,
                     rate_source='v_dot')

        p4.add_parameter('h', val=0.0, opt=False, units='m')
        p4.add_parameter('T', val=120101.98/2, opt=False, units='N')
        p4.add_parameter('mu_r', val=0.03, opt=False, units=None)

        p4.add_polynomial_control('alpha', order=1, opt=True, units='deg', shape=(1,), lower=0, upper=10, ref=10)

        # p4.add_control('alpha', val=0.0, opt=True, lower=0, upper=10, units='deg')

        p4.add_timeseries_output('*')

        p4.add_boundary_constraint('F_r', loc='final', equals=0, ref=1000)
        # p4.add_boundary_constraint('alpha', loc='final', equals=10, units='deg')

        # p4.add_path_constraint('alpha_rate', lower=0, upper=3, units='deg/s')

        p.model.linear_solver = om.DirectSolver()
        
        #
        # Link the phases
        #
        traj.link_phases(['brake_release_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['brake_release_to_v1', 'rto'], vars=['time', 'r', 'v', 'alpha'])

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p.set_val('traj.brake_release_to_v1.t_initial', 0)
        p.set_val('traj.brake_release_to_v1.t_duration', 35)

        p.set_val('traj.brake_release_to_v1.states:r', p1.interpolate(ys=[0, 2500.0], nodes='state_input'))
        p.set_val('traj.brake_release_to_v1.states:v', p1.interpolate(ys=[0, 100.0], nodes='state_input'))

        p.set_val('traj.brake_release_to_v1.parameters:alpha', 0, units='deg')
        p.set_val('traj.brake_release_to_v1.parameters:h', 0.0)
        
        #
        
        p.set_val('traj.v1_to_vr.t_initial', 35)
        p.set_val('traj.v1_to_vr.t_duration', 35)

        p.set_val('traj.v1_to_vr.states:r', p1.interpolate(ys=[2500, 300.0], nodes='state_input'))
        p.set_val('traj.v1_to_vr.states:v', p1.interpolate(ys=[100, 110.0], nodes='state_input'))

        p.set_val('traj.v1_to_vr.parameters:alpha', 0.0, units='deg')

        p.set_val('traj.v1_to_vr.parameters:h', 0.0)
        
        #
        
        p.set_val('traj.rto.t_initial', 35)
        p.set_val('traj.rto.t_duration', 1)

        p.set_val('traj.rto.states:r', p1.interpolate(ys=[2500, 5000.0], nodes='state_input'))
        p.set_val('traj.rto.states:v', p1.interpolate(ys=[110, 0.0], nodes='state_input'))

        p.set_val('traj.rto.parameters:alpha', 0.0, units='deg')
        p.set_val('traj.rto.parameters:h', 0.0)
        p.set_val('traj.rto.parameters:T', 0.0)
        p.set_val('traj.rto.parameters:mu_r', 0.3)

        #

        p.set_val('traj.rotate.t_initial', 35)
        p.set_val('traj.rotate.t_duration', 35)

        p.set_val('traj.rotate.states:r', p1.interpolate(ys=[5000, 5500.0], nodes='state_input'))
        p.set_val('traj.rotate.states:v', p1.interpolate(ys=[160, 170.0], nodes='state_input'))

        p.set_val('traj.rotate.controls:alpha', 0.0, units='deg')

        p.set_val('traj.rotate.parameters:h', 0.0)

        #
        # Solve for the optimal trajectory
        #
        p.run_driver()

        #
        # Get the explicitly simulated solution and plot the results
        #
        exp_out = traj.simulate()


        fig, axes = plt.subplots(3, 1)

        for phase_name in ['brake_release_to_v1', 'v1_to_vr', 'rotate', 'rto']:

            axes[0].plot(p.get_val(f'traj.{phase_name}.timeseries.time'),
                         p.get_val(f'traj.{phase_name}.timeseries.states:r', units='ft'), 'o')

            axes[0].plot(exp_out.get_val(f'traj.{phase_name}.timeseries.time'),
                         exp_out.get_val(f'traj.{phase_name}.timeseries.states:r', units='ft'), '-')

            axes[1].plot(p.get_val(f'traj.{phase_name}.timeseries.time'),
                         p.get_val(f'traj.{phase_name}.timeseries.states:v', units='kn'), 'o')

            axes[1].plot(exp_out.get_val(f'traj.{phase_name}.timeseries.time'),
                         exp_out.get_val(f'traj.{phase_name}.timeseries.states:v', units='kn'), '-')

            axes[2].plot(p.get_val(f'traj.{phase_name}.timeseries.time'),
                         p.get_val(f'traj.{phase_name}.timeseries.F_r', units='N'), 'o')

            axes[2].plot(exp_out.get_val(f'traj.{phase_name}.timeseries.time'),
                         exp_out.get_val(f'traj.{phase_name}.timeseries.F_r', units='N'), '-')

        print(p.get_val('traj.v1_to_vr.timeseries.v_over_v_stall'))

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
