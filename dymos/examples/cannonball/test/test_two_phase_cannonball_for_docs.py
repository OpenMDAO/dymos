from __future__ import print_function, division, absolute_import

import os
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TestTwoPhaseCannonballForDocs(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['ex_two_phase_cannonball.db', 'ex_two_phase_cannonball_sim.db',
                         'coloring.json']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_two_phase_cannonball_for_docs(self):
        from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, SqliteRecorder, \
            pyOptSparseDriver
        from openmdao.utils.assert_utils import assert_rel_error

        from dymos import Phase, Trajectory
        from dymos.examples.cannonball.cannonball_ode import CannonballODE

        from dymos.examples.cannonball.size_comp import CannonballSizeComp

        p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['dynamic_simul_derivs'] = True

        external_params = p.model.add_subsystem('external_params', IndepVarComp())

        external_params.add_output('radius', val=0.10, units='m')
        external_params.add_output('dens', val=7.87, units='g/cm**3')

        external_params.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10)

        p.model.add_subsystem('size_comp', CannonballSizeComp())

        traj = p.model.add_subsystem('traj', Trajectory())

        # First Phase (ascent)
        ascent = Phase('radau-ps',
                       ode_class=CannonballODE,
                       num_segments=5,
                       transcription_order=3,
                       compressed=True)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100)
        ascent.set_state_options('r', fix_initial=True, fix_final=False)
        ascent.set_state_options('h', fix_initial=True, fix_final=False)
        ascent.set_state_options('gam', fix_initial=False, fix_final=True)
        ascent.set_state_options('v', fix_initial=False, fix_final=False)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('kinetic_energy.ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000)

        # Second Phase (descent)
        descent = Phase('gauss-lobatto',
                        ode_class=CannonballODE,
                        num_segments=5,
                        transcription_order=3,
                        compressed=True)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100)
        descent.set_state_options('r', fix_initial=False, fix_final=False)
        descent.set_state_options('h', fix_initial=False, fix_final=True)
        descent.set_state_options('gam', fix_initial=False, fix_final=False)
        descent.set_state_options('v', fix_initial=False, fix_final=False)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_design_parameter('CD', val=0.5, units=None, opt=False)
        traj.add_design_parameter('CL', val=0.0, units=None, opt=False)
        traj.add_design_parameter('T', val=0.0, units='N', opt=False)
        traj.add_design_parameter('alpha', val=0.0, units='deg', opt=False)

        # Add externally-provided design parameters to the trajectory.
        traj.add_input_parameter('mass',
                                 target_params={'ascent': 'm', 'descent': 'm'},
                                 val=1.0,
                                 units='kg')

        traj.add_input_parameter('S', val=0.005, units='m**2')

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.input_parameters:mass')
        p.model.connect('size_comp.S', 'traj.input_parameters:S')

        # Finish Problem Setup
        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.driver.add_recorder(SqliteRecorder('ex_two_phase_cannonball.db'))

        p.setup(check=True)

        # Set Initial Guesses
        p.set_val('external_params.radius', 0.05, units='m')
        p.set_val('external_params.dens', 7.87, units='g/cm**3')

        p.set_val('traj.design_parameters:CD', 0.5)
        p.set_val('traj.design_parameters:CL', 0.0)
        p.set_val('traj.design_parameters:T', 0.0)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interpolate(ys=[0, 100], nodes='state_input'))
        p.set_val('traj.ascent.states:h', ascent.interpolate(ys=[0, 100], nodes='state_input'))
        p.set_val('traj.ascent.states:v', ascent.interpolate(ys=[200, 150], nodes='state_input'))
        p.set_val('traj.ascent.states:gam', ascent.interpolate(ys=[25, 0], nodes='state_input'),
                  units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interpolate(ys=[100, 200], nodes='state_input'))
        p.set_val('traj.descent.states:h', descent.interpolate(ys=[100, 0], nodes='state_input'))
        p.set_val('traj.descent.states:v', descent.interpolate(ys=[150, 200], nodes='state_input'))
        p.set_val('traj.descent.states:gam', descent.interpolate(ys=[0, -45], nodes='state_input'),
                  units='deg')

        p.run_driver()

        assert_rel_error(self, p.get_val('traj.descent.states:r')[-1],
                         3191.83945861, tolerance=1.0E-2)

        exp_out = traj.simulate(times=100, record_file='ex_two_phase_cannonball_sim.db')

        # exp_out_loaded = load_simulation_results('ex_two_phase_cannonball_sim.db')

        print('optimal radius: {0:6.4f} m '.format(p.get_val('external_params.radius',
                                                             units='m')[0]))
        print('cannonball mass: {0:6.4f} kg '.format(p.get_val('size_comp.mass',
                                                               units='kg')[0]))
        print('launch angle: {0:6.4f} '
              'deg '.format(p.get_val('traj.ascent.timeseries.states:gam',  units='deg')[0, 0]))
        print('maximum range: {0:6.4f} '
              'm '.format(p.get_val('traj.descent.timeseries.states:r')[-1, 0]))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        time_imp = {'ascent': p.get_val('traj.ascent.timeseries.time'),
                    'descent': p.get_val('traj.descent.timeseries.time')}

        time_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.time'),
                    'descent': exp_out.get_val('traj.descent.timeseries.time')}

        r_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:r'),
                 'descent': p.get_val('traj.descent.timeseries.states:r')}

        r_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:r'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:r')}

        h_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:h'),
                 'descent': p.get_val('traj.descent.timeseries.states:h')}

        h_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:h'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:h')}

        axes.plot(r_imp['ascent'], h_imp['ascent'], 'bo')

        axes.plot(r_imp['descent'], h_imp['descent'], 'ro')

        axes.plot(r_exp['ascent'], h_exp['ascent'], 'b--')

        axes.plot(r_exp['descent'], h_exp['descent'], 'r--')

        axes.set_xlabel('range (m)')
        axes.set_ylabel('altitude (m)')

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
        states = ['r', 'h', 'v', 'gam']
        for i, state in enumerate(states):
            x_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:{0}'.format(state)),
                     'descent': p.get_val('traj.descent.timeseries.states:{0}'.format(state))}

            x_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:{0}'.format(state)),
                     'descent': exp_out.get_val('traj.descent.timeseries.states:{0}'.format(state))}

            axes[i].set_ylabel(state)

            axes[i].plot(time_imp['ascent'], x_imp['ascent'], 'bo')
            axes[i].plot(time_imp['descent'], x_imp['descent'], 'ro')
            axes[i].plot(time_exp['ascent'], x_exp['ascent'], 'b--')
            axes[i].plot(time_exp['descent'], x_exp['descent'], 'r--')

        params = ['CL', 'CD', 'T', 'alpha', 'm', 'S']
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6))
        for i, param in enumerate(params):
            p_imp = {
                'ascent': p.get_val('traj.ascent.timeseries.traj_parameters:{0}'.format(param)),
                'descent': p.get_val('traj.descent.timeseries.traj_parameters:{0}'.format(param))}

            p_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.'
                                               'traj_parameters:{0}'.format(param)),
                     'descent': exp_out.get_val('traj.descent.timeseries.'
                                                'traj_parameters:{0}'.format(param))}

            axes[i].set_ylabel(param)

            axes[i].plot(time_imp['ascent'], p_imp['ascent'], 'bo')
            axes[i].plot(time_imp['descent'], p_imp['descent'], 'ro')
            axes[i].plot(time_exp['ascent'], p_exp['ascent'], 'b--')
            axes[i].plot(time_exp['descent'], p_exp['descent'], 'r--')

        plt.show()
