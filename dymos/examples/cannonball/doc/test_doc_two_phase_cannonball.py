import unittest

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestTwoPhaseCannonballForDocs(unittest.TestCase):

    @save_for_docs
    def test_two_phase_cannonball_for_docs(self):
        import numpy as np
        from scipy.interpolate import interp1d

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.models.atmosphere.atmos_1976 import USatm1976Data

        #############################################
        # Component for the design part of the model
        #############################################
        class CannonballSizeComp(om.ExplicitComponent):
            """
            Compute the area and mass of a cannonball with a given radius and density.

            Notes
            -----
            This component is not vectorized with 'num_nodes' as is the usual way
            with Dymos, but is instead intended to compute a scalar mass and reference
            area from scalar radius and density inputs. This component does not reside
            in the ODE but instead its outputs are connected to the trajectory via
            input design parameters.
            """
            def setup(self):
                self.add_input(name='radius', val=1.0, desc='cannonball radius', units='m')
                self.add_input(name='dens', val=7870., desc='cannonball density', units='kg/m**3')

                self.add_output(name='mass', shape=(1,), desc='cannonball mass', units='kg')
                self.add_output(name='S', shape=(1,), desc='aerodynamic reference area', units='m**2')

                self.declare_partials(of='mass', wrt='dens')
                self.declare_partials(of='mass', wrt='radius')

                self.declare_partials(of='S', wrt='radius')

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                dens = inputs['dens']

                outputs['mass'] = (4/3.) * dens * np.pi * radius ** 3
                outputs['S'] = np.pi * radius ** 2

            def compute_partials(self, inputs, partials):
                radius = inputs['radius']
                dens = inputs['dens']

                partials['mass', 'dens'] = (4/3.) * np.pi * radius ** 3
                partials['mass', 'radius'] = 4. * dens * np.pi * radius ** 2

                partials['S', 'radius'] = 2 * np.pi * radius

        #############################################
        # Build the ODE class
        #############################################
        class CannonballODE(om.ExplicitComponent):
            """
            Cannonball ODE assuming flat earth and accounting for air resistance
            """

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                # static parameters
                self.add_input('m', units='kg')
                self.add_input('S', units='m**2')
                # 0.5 good assumption for a sphere
                self.add_input('CD', 0.5)

                # time varying inputs
                self.add_input('h', units='m', shape=nn)
                self.add_input('v', units='m/s', shape=nn)
                self.add_input('gam', units='rad', shape=nn)

                # state rates
                self.add_output('v_dot', shape=nn, units='m/s**2', tags=['state_rate_source:v'])
                self.add_output('gam_dot', shape=nn, units='rad/s', tags=['state_rate_source:gam'])
                self.add_output('h_dot', shape=nn, units='m/s', tags=['state_rate_source:h'])
                self.add_output('r_dot', shape=nn, units='m/s', tags=['state_rate_source:r'])
                self.add_output('ke', shape=nn, units='J')

                # Ask OpenMDAO to compute the partial derivatives using complex-step
                # with a partial coloring algorithm for improved performance, and use
                # a graph coloring algorithm to automatically detect the sparsity pattern.
                self.declare_coloring(wrt='*', method='cs')

                alt_data = USatm1976Data.alt * om.unit_conversion('ft', 'm')[0]
                rho_data = USatm1976Data.rho * om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
                self.rho_interp = interp1d(np.array(alt_data, dtype=complex),
                                           np.array(rho_data, dtype=complex),
                                           kind='linear')

            def compute(self, inputs, outputs):

                gam = inputs['gam']
                v = inputs['v']
                h = inputs['h']
                m = inputs['m']
                S = inputs['S']
                CD = inputs['CD']

                GRAVITY = 9.80665  # m/s**2

                # handle complex-step gracefully from the interpolant
                if np.iscomplexobj(h):
                    rho = self.rho_interp(inputs['h'])
                else:
                    rho = self.rho_interp(inputs['h']).real

                q = 0.5*rho*inputs['v']**2
                qS = q * S
                D = qS * CD
                cgam = np.cos(gam)
                sgam = np.sin(gam)
                outputs['v_dot'] = - D/m-GRAVITY*sgam
                outputs['gam_dot'] = -(GRAVITY/v)*cgam
                outputs['h_dot'] = v*sgam
                outputs['r_dot'] = v*cgam
                outputs['ke'] = 0.5*m*v**2

        #############################################
        # Setup the Dymos problem
        #############################################

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        p.model.add_subsystem('size_comp', CannonballSizeComp(),
                              promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10,
                               ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero
        # so that the phase ends at apogee).
        # The output of the ODE which provides the rate source for each state
        # is obtained from the tags used on those outputs in the ODE.
        # The units of the states are automatically inferred by multiplying the units
        # of those rates by the time units.
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100),
                                duration_ref=100, units='s')
        ascent.set_state_options('r', fix_initial=True, fix_final=False)
        ascent.set_state_options('h', fix_initial=True, fix_final=False)
        ascent.set_state_options('gam', fix_initial=False, fix_final=True)
        ascent.set_state_options('v', fix_initial=False, fix_final=False)

        ascent.add_parameter('S', units='m**2', dynamic=False)
        ascent.add_parameter('m', units='kg', dynamic=False)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial',
                                       upper=400000, lower=0, ref=100000)

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free, since
        #    they will be linked to the final states of ascent.
        # Final altitude is fixed, because we will set
        #    it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r')
        descent.add_state('h', fix_initial=False, fix_final=True)
        descent.add_state('gam', fix_initial=False, fix_final=False)
        descent.add_state('v', fix_initial=False, fix_final=False)

        descent.add_parameter('S', units='m**2', dynamic=False)
        descent.add_parameter('m', units='kg', dynamic=False)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False, dynamic=False)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters
        # named 'mass' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'mass', 'descent': 'mass'}, dynamic=False)

        # In this case, by omitting targets, we're connecting these
        # parameters to parameters with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005, dynamic=False)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        # Issue Connections
        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # A linear solver at the top level can improve performance.
        p.model.linear_solver = om.DirectSolver()

        # Finish Problem Setup
        p.setup()

        #############################################
        # Set constants and initial guesses
        #############################################
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        p.set_val('traj.parameters:CD', 0.5)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interpolate(ys=[0, 100],
                  nodes='state_input'))
        p.set_val('traj.ascent.states:h', ascent.interpolate(ys=[0, 100],
                  nodes='state_input'))
        p.set_val('traj.ascent.states:v', ascent.interpolate(ys=[200, 150],
                  nodes='state_input'))
        p.set_val('traj.ascent.states:gam', ascent.interpolate(ys=[25, 0],
                  nodes='state_input'), units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interpolate(ys=[100, 200],
                  nodes='state_input'))
        p.set_val('traj.descent.states:h', descent.interpolate(ys=[100, 0],
                  nodes='state_input'))
        p.set_val('traj.descent.states:v', descent.interpolate(ys=[150, 200],
                  nodes='state_input'))
        p.set_val('traj.descent.states:gam', descent.interpolate(ys=[0, -45],
                  nodes='state_input'), units='deg')

        #####################################################
        # Run the optimization and final explicit simulation
        #####################################################
        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)

        # use the explicit simulation to check the final collocation solution accuracy
        exp_out = traj.simulate()

        #############################################
        # Plot the results
        #############################################
        rad = p.get_val('radius', units='m')[0]
        print(f'optimal radius: {rad} m ')
        mass = p.get_val('size_comp.mass', units='kg')[0]
        print(f'cannonball mass: {mass} kg ')
        angle = p.get_val('traj.ascent.timeseries.states:gam', units='deg')[0, 0]
        print(f'launch angle: {angle} deg')
        max_range = p.get_val('traj.descent.timeseries.states:r')[-1, 0]
        print(f'maximum range: {max_range} m')

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
            x_imp = {'ascent': p.get_val(f'traj.ascent.timeseries.states:{state}'),
                     'descent': p.get_val(f'traj.descent.timeseries.states:{state}')}

            x_exp = {'ascent': exp_out.get_val(f'traj.ascent.timeseries.states:{state}'),
                     'descent': exp_out.get_val(f'traj.descent.timeseries.states:{state}')}

            axes[i].set_ylabel(state)

            axes[i].plot(time_imp['ascent'], x_imp['ascent'], 'bo')
            axes[i].plot(time_imp['descent'], x_imp['descent'], 'ro')
            axes[i].plot(time_exp['ascent'], x_exp['ascent'], 'b--')
            axes[i].plot(time_exp['descent'], x_exp['descent'], 'r--')

        params = ['m', 'S']
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6))
        for i, param in enumerate(params):
            p_imp = {
                'ascent': p.get_val(f'traj.ascent.timeseries.parameters:{param}'),
                'descent': p.get_val(f'traj.descent.timeseries.parameters:{param}')}

            p_exp = {'ascent': exp_out.get_val(f'traj.ascent.timeseries.parameters:{param}'),
                     'descent': exp_out.get_val(f'traj.descent.timeseries.parameters:{param}')}

            axes[i].set_ylabel(param)

            axes[i].plot(time_imp['ascent'], p_imp['ascent'], 'bo')
            axes[i].plot(time_imp['descent'], p_imp['descent'], 'ro')
            axes[i].plot(time_exp['ascent'], p_exp['ascent'], 'b--')
            axes[i].plot(time_exp['descent'], p_exp['descent'], 'r--')

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
