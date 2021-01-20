import unittest

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestTwoPhaseCannonballForDocs(unittest.TestCase):

    @save_for_docs
    def test_two_phase_cannonball_for_docs(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.cannonball.size_comp import CannonballSizeComp
        from dymos.examples.cannonball.cannonball_phase import CannonballPhase

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        external_params = p.model.add_subsystem('external_params', om.IndepVarComp())

        external_params.add_output('radius', val=0.10, units='m')
        external_params.add_output('dens', val=7.87, units='g/cm**3')

        external_params.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10)

        p.model.add_subsystem('size_comp', CannonballSizeComp())

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = CannonballPhase(transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.set_state_options('r', fix_initial=True, fix_final=False)
        ascent.set_state_options('h', fix_initial=True, fix_final=False)
        ascent.set_state_options('gam', fix_initial=False, fix_final=True)
        ascent.set_state_options('v', fix_initial=False, fix_final=False)

        ascent.add_parameter('S', targets=['aero.S'], units='m**2')
        ascent.add_parameter('mass', targets=['eom.m', 'kinetic_energy.m'], units='kg')

        # Limit the muzzle energy
        ascent.add_boundary_constraint('kinetic_energy.ke', loc='initial',
                                       upper=400000, lower=0, ref=100000)

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = CannonballPhase(transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', )
        descent.add_state('h', fix_initial=False, fix_final=True)
        descent.add_state('gam', fix_initial=False, fix_final=False)
        descent.add_state('v', fix_initial=False, fix_final=False)

        descent.add_parameter('S', targets=['aero.S'], units='m**2')
        descent.add_parameter('mass', targets=['eom.m', 'kinetic_energy.m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['aero.CD'], 'descent': ['aero.CD']},
                           val=0.5, units=None, opt=False)
        traj.add_parameter('CL',
                           targets={'ascent': ['aero.CL'], 'descent': ['aero.CL']},
                           val=0.0, units=None, opt=False)
        traj.add_parameter('T',
                           targets={'ascent': ['eom.T'], 'descent': ['eom.T']},
                           val=0.0, units='N', opt=False)
        traj.add_parameter('alpha',
                           targets={'ascent': ['eom.alpha'], 'descent': ['eom.alpha']},
                           val=0.0, units='deg', opt=False)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'mass', 'descent': 'mass'})

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # A linear solver at the top level can improve performance.
        p.model.linear_solver = om.DirectSolver()

        # Finish Problem Setup
        p.setup()

        # Set Initial Guesses
        p.set_val('external_params.radius', 0.05, units='m')
        p.set_val('external_params.dens', 7.87, units='g/cm**3')

        p.set_val('traj.parameters:CD', 0.5)
        p.set_val('traj.parameters:CL', 0.0)
        p.set_val('traj.parameters:T', 0.0)

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

        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)

        exp_out = traj.simulate()

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

        params = ['CL', 'CD', 'T', 'alpha', 'mass', 'S']
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6))
        for i, param in enumerate(params):
            p_imp = {
                'ascent': p.get_val('traj.ascent.timeseries.parameters:{0}'.format(param)),
                'descent': p.get_val('traj.descent.timeseries.parameters:{0}'.format(param))}

            p_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.'
                                               'parameters:{0}'.format(param)),
                     'descent': exp_out.get_val('traj.descent.timeseries.'
                                                'parameters:{0}'.format(param))}

            axes[i].set_ylabel(param)

            axes[i].plot(time_imp['ascent'], p_imp['ascent'], 'bo')
            axes[i].plot(time_imp['descent'], p_imp['descent'], 'ro')
            axes[i].plot(time_exp['ascent'], p_exp['ascent'], 'b--')
            axes[i].plot(time_exp['descent'], p_exp['descent'], 'r--')

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
