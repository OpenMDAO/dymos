import unittest
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal


@use_tempdirs
class TestCannonballForJOSS(unittest.TestCase):

    def test_results(self):
        # begin code for paper
        import numpy as np
        from scipy.interpolate import interp1d
        import matplotlib.pyplot as plt

        import openmdao.api as om

        import dymos as dm
        from dymos.models.atmosphere.atmos_1976 import USatm1976Data as USatm1976Data

        # CREATE an atmosphere interpolant
        english_to_metric_rho = om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
        english_to_metric_alt = om.unit_conversion('ft', 'm')[0]
        rho_interp = interp1d(np.array(USatm1976Data.alt * english_to_metric_alt, dtype=complex),
                              np.array(USatm1976Data.rho * english_to_metric_rho, dtype=complex),
                              kind='linear')


        class CannonballSize(om.ExplicitComponent):
            """
            Static calculations performed before the dynamic model
            """

            def setup(self):
                self.add_input(name='radius', val=1.0,
                               desc='cannonball radius', units='m')
                self.add_input(name='density', val=7870.,
                               desc='cannonball density', units='kg/m**3')

                self.add_output(name='mass', shape=(1,),
                                desc='cannonball mass', units='kg')
                self.add_output(name='area', shape=(1,),
                                desc='aerodynamic reference area', units='m**2')

                self.declare_partials(of='*', wrt='*', method='cs')

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                rho = inputs['density']

                outputs['mass'] = (4 / 3.) * rho * np.pi * radius ** 3
                outputs['area'] = np.pi * radius ** 2


        class CannonballODE(om.ExplicitComponent):
            """
            Cannonball ODE assuming flat earth and accounting for air resistance
            """

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                # static parameters
                self.add_input('mass', units='kg')
                self.add_input('area', units='m**2')

                # time varying inputs
                self.add_input('alt', units='m', shape=nn)
                self.add_input('v', units='m/s', shape=nn)
                self.add_input('gam', units='rad', shape=nn)

                # state rates
                self.add_output('v_dot', shape=nn, units='m/s**2')
                self.add_output('gam_dot', shape=nn, units='rad/s')
                self.add_output('h_dot', shape=nn, units='m/s')
                self.add_output('r_dot', shape=nn, units='m/s')
                self.add_output('ke', shape=nn, units='J')

                # Ask OpenMDAO to compute the partial derivatives using complex-step
                # with a partial coloring algorithm for improved performance
                self.declare_partials('*', '*', method='cs')
                self.declare_coloring(wrt='*', method='cs')

            def compute(self, inputs, outputs):

                gam = inputs['gam']
                v = inputs['v']
                alt = inputs['alt']
                m = inputs['mass']
                S = inputs['area']

                CD = 0.5  # good assumption for a sphere
                GRAVITY = 9.80665  # m/s**2

                # handle complex-step gracefully from the interpolant
                if np.iscomplexobj(alt):
                    rho = rho_interp(inputs['alt'])
                else:
                    rho = rho_interp(inputs['alt']).real

                q = 0.5 * rho * inputs['v'] ** 2
                qS = q * S
                D = qS * CD
                cgam = np.cos(gam)
                sgam = np.sin(gam)
                outputs['v_dot'] = - D / m - GRAVITY * sgam
                outputs['gam_dot'] = -(GRAVITY / v) * cgam
                outputs['h_dot'] = v * sgam
                outputs['r_dot'] = v * cgam
                outputs['ke'] = 0.5 * m * v ** 2

        p = om.Problem()

        ###################################
        # Co-design part of the model,
        # static analysis outside of Dymos
        ###################################
        static_calcs = p.model.add_subsystem('static_calcs', CannonballSize())
        static_calcs.add_design_var('radius', lower=0.01, upper=0.10,
                                    ref0=0.01, ref=0.10)

        p.model.connect('static_calcs.mass', 'traj.parameters:mass')
        p.model.connect('static_calcs.area', 'traj.parameters:area')

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        # Declare parameters that will be constant across
        # the two phases of the trajectory, so we can connect to it only once
        traj.add_parameter('mass', units='kg', val=0.01, static_target=True)
        traj.add_parameter('area', units='m**2', static_target=True)

        tx = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = dm.Phase(transcription=tx, ode_class=CannonballODE)
        traj.add_phase('ascent', ascent)

        ###################################
        # first phase: ascent
        ###################################
        # All initial states except flight path angle are fixed
        ascent.add_state('r', units='m', rate_source='r_dot',
                         fix_initial=True, fix_final=False)
        ascent.add_state('h', units='m', rate_source='h_dot',
                         fix_initial=True, fix_final=False)
        ascent.add_state('v', units='m/s', rate_source='v_dot',
                         fix_initial=False, fix_final=False)
        # Final flight path angle is fixed (
        #     we will set it to zero so that the phase ends at apogee)
        ascent.add_state('gam', units='rad', rate_source='gam_dot',
                         fix_initial=False, fix_final=True)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100),
                                duration_ref=100, units='s')

        ascent.add_parameter('mass', units='kg', val=0.01, static_target=True)
        ascent.add_parameter('area', units='m**2', static_target=True)

        # Limit the initial muzzle energy to create a well posed problem
        # with respect to cannonball size and initial velocity
        ascent.add_boundary_constraint('ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000)

        ###################################
        # second phase: descent
        ###################################
        tx = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(transcription=tx, ode_class=CannonballODE)
        traj.add_phase('descent', descent)

        # All initial states and time are free so their
        #    values can be linked to the final ascent values
        # Final altitude is fixed to 0 to ensure final impact on the ground
        descent.add_state('r', units='m', rate_source='r_dot',
                          fix_initial=False, fix_final=False)
        descent.add_state('h', units='m', rate_source='h_dot',
                          fix_initial=False, fix_final=True)
        descent.add_state('gam', units='rad', rate_source='gam_dot',
                          fix_initial=False, fix_final=False)
        descent.add_state('v', units='m/s', rate_source='v_dot',
                          fix_initial=False, fix_final=False)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')

        descent.add_parameter('mass', units='kg', val=0.01, static_target=True)
        descent.add_parameter('area', units='m**2', static_target=True)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        # maximize range
        descent.add_objective('r', loc='final', ref=-1.0)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        # Finish Problem Setup
        p.setup()

        # Set Initial guesses for static dvs and ascent
        p.set_val('static_calcs.radius', 0.05, units='m')
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        # more intitial guesses for descent
        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')

        dm.run_problem(p, simulate=True, make_plots=True)

        fig, ax = plt.subplots()
        x0 = p.get_val('traj.ascent.timeseries.states:r', units='m')
        y0 = p.get_val('traj.ascent.timeseries.states:h', units='m')
        x1 = p.get_val('traj.descent.timeseries.states:r', units='m')
        y1 = p.get_val('traj.descent.timeseries.states:h', units='m')
        tab20 = plt.cm.get_cmap('tab20').colors
        ax.plot(x0, y0, marker='o', label='ascent', color=tab20[0])
        ax.plot(x1, y1, marker='o', label='descent', color=tab20[1])
        ax.legend(loc='best')
        ax.set_xlabel('range (m)')
        ax.set_ylabel('height (m)')
        fig.savefig('cannonball_hr.png', bbox_inches='tight')
        # End code for paper

        assert_near_equal(x1[-1], 3064, tolerance=1.0E-4)
