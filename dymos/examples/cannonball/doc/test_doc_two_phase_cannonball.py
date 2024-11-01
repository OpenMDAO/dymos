import unittest

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


@use_tempdirs
class TestTwoPhaseCannonballForDocs(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_two_phase_cannonball_for_docs(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.components.interp_util.interp import InterpND
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
                self.add_output('v_dot', shape=nn, units='m/s**2', tags=['dymos.state_rate_source:v'])
                self.add_output('gam_dot', shape=nn, units='rad/s', tags=['dymos.state_rate_source:gam'])
                self.add_output('h_dot', shape=nn, units='m/s', tags=['dymos.state_rate_source:h'])
                self.add_output('r_dot', shape=nn, units='m/s', tags=['dymos.state_rate_source:r'])
                self.add_output('ke', shape=nn, units='J')

                # Ask OpenMDAO to compute the partial derivatives using finite-difference
                # with a partial coloring algorithm for improved performance, and use
                # a graph coloring algorithm to automatically detect the sparsity pattern.
                self.declare_coloring(wrt='*', method='fd')

                alt_data = USatm1976Data.alt * om.unit_conversion('ft', 'm')[0]
                rho_data = USatm1976Data.rho * om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
                self.rho_interp = InterpND(points=np.array(alt_data),
                                           values=np.array(rho_data),
                                           method='slinear')

            def compute(self, inputs, outputs):

                gam = inputs['gam']
                v = inputs['v']
                h = inputs['h']
                m = inputs['m']
                S = inputs['S']
                CD = inputs['CD']

                GRAVITY = 9.80665  # m/s**2

                rho = self.rho_interp.interpolate(inputs['h'])

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

        ascent.add_parameter('S', units='m**2', static_target=True)
        ascent.add_parameter('m', units='kg', static_target=True)

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

        descent.add_parameter('S', units='m**2', static_target=True)
        descent.add_parameter('m', units='kg', static_target=True)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False, static_target=True)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters
        # named 'm' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'm', 'descent': 'm'}, static_target=True)

        # In this case, by omitting targets, we're connecting these
        # parameters to parameters with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005, static_target=True)

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

        traj.set_parameter_val('CD', 0.5)

        ascent.set_time_val(initial=0.0, duration=10.0)

        ascent.set_state_val('r', [0, 100])
        ascent.set_state_val('h', [0, 100])
        ascent.set_state_val('v', [200, 150])
        ascent.set_state_val('gam', [25, 0], units='deg')

        descent.set_time_val(initial=10.0, duration=10.0)

        descent.set_state_val('r', [100, 200])
        descent.set_state_val('h', [100, 0])
        descent.set_state_val('v', [150, 200])
        descent.set_state_val('gam', [0, -45], units='deg')

        #####################################################
        # Run the optimization and final explicit simulation
        #####################################################
        dm.run_problem(p, simulate=True)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
