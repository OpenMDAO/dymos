import unittest

import numpy as np

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


def initial_guess(t_dur, gam0=np.pi/3):
    t = np.linspace(0, t_dur, num=100)
    g = -9.80665
    v0 = -0.5*g*t_dur/np.sin(gam0)
    r = v0*np.cos(gam0)*t
    h = v0*np.sin(gam0)*t + 0.5*g*t**2
    v = np.sqrt(v0*np.cos(gam0)**2 + (v0*np.sin(gam0) + g*t)**2)
    gam = np.arctan2(v0*np.sin(gam0) + g*t, v0*np.cos(gam0)**2)

    guess = {'t': t, 'r': r, 'h': h, 'v': v, 'gam': gam}

    return guess


@use_tempdirs
class TestCannonballImplicitDuration(unittest.TestCase):

    def test_cannonball_implicit_duration(self):
        import openmdao.api as om
        from scipy.interpolate import make_interp_spline
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
                self.rho_interp = make_interp_spline(alt_data, rho_data, k=1)

            def compute(self, inputs, outputs):

                gam = inputs['gam']
                v = inputs['v']
                h = inputs['h']
                m = inputs['m']
                S = inputs['S']
                CD = inputs['CD']

                GRAVITY = 9.80665  # m/s**2

                rho = self.rho_interp(h)

                q = 0.5*rho*v**2
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

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        p.model.add_subsystem('size_comp', CannonballSizeComp(),
                              promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10,
                               ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=20, order=3, compressed=False)
        phase = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        phase = traj.add_phase('phase', phase)

        # All initial states except flight path angle are fixed
        # The output of the ODE which provides the rate source for each state
        # is obtained from the tags used on those outputs in the ODE.
        # The units of the states are automatically inferred by multiplying the units
        # of those rates by the time units.
        phase.set_time_options(fix_initial=True, duration_bounds=(1, 100), units='s')
        phase.add_state('r', fix_initial=True, solve_segments='forward')
        phase.add_state('h', fix_initial=True, solve_segments='forward')
        phase.add_state('gam', fix_initial=False, initial_bounds=(0, np.pi/2), solve_segments='forward')
        phase.add_state('v', fix_initial=False, solve_segments='forward')

        phase.add_parameter('S', units='m**2', static_target=True)
        phase.add_parameter('m', units='kg', static_target=True)
        phase.add_parameter('CD', val=0.5, opt=False, static_target=True)

        phase.add_boundary_constraint('ke', loc='initial',
                                      upper=400000, lower=0, ref=100000)

        # A duration balance is added setting altitude to zero.
        # A nonlinear solver is used to find the duration of required to satisfy the condition.
        # The duration was bounded to be greater than 1 to ensure the solver did not
        # converge to the initial point.
        phase.set_duration_balance('h', val=0.0)

        # In this problem, the default ArmijoGoldsteinLS has issues with extrapolating
        # the states and causes the optimization to fail.
        # Using the default linesearch or BoundsEnforceLS work better here.
        phase.nonlinear_solver = om.NewtonSolver(iprint=0, solve_subsystems=True,
                                                 maxiter=100, stall_limit=3)
        phase.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        phase.linear_solver = om.DirectSolver()

        phase.add_objective('r', loc='final', scaler=-1.0)

        p.model.connect('size_comp.mass', 'traj.phase.parameters:m')
        p.model.connect('size_comp.S', 'traj.phase.parameters:S')

        # Finish Problem Setup
        p.setup()

        #############################################
        # Set constants and initial guesses
        #############################################
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        p.set_val('traj.phase.parameters:CD', 0.5)

        guess = initial_guess(t_dur=30.0)

        phase.set_time_val(initial=0.0, duration=30.0)

        phase.set_state_val('r', vals=guess['r'], time_vals=guess['t'])
        phase.set_state_val('h', vals=guess['h'], time_vals=guess['t'])
        phase.set_state_val('v', vals=guess['v'], time_vals=guess['t'])
        phase.set_state_val('gam', vals=guess['gam'], time_vals=guess['t'], units='rad')

        #####################################################
        # Run the optimization and final explicit simulation
        #####################################################
        dm.run_problem(p, simulate=True)

        assert_near_equal(p.get_val('traj.phase.states:r')[-1],
                          3183.25, tolerance=1.0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
