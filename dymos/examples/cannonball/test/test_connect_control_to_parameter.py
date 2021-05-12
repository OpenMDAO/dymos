import unittest

import numpy as np

import openmdao
import openmdao.api as om

from openmdao.utils.testing_utils import use_tempdirs

from dymos.examples.cannonball.cannonball_ode import rho_interp

om_dev_version = openmdao.__version__.endswith('dev')
om_version = tuple(int(s) for s in openmdao.__version__.split('-')[0].split('.'))


class CannonballODEVectorCD(om.ExplicitComponent):
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

        # This will be used as both a control and a parameter
        self.add_input('CD', 0.5, shape=nn)

        # time varying inputs
        self.add_input('h', units='m', shape=nn)
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
        h = inputs['h']
        m = inputs['m']
        S = inputs['S']
        CD = inputs['CD']

        GRAVITY = 9.80665  # m/s**2

        # handle complex-step gracefully from the interpolant
        if np.iscomplexobj(h):
            rho = rho_interp(inputs['h'])
        else:
            rho = rho_interp(inputs['h']).real

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


@use_tempdirs
class TestConnectControlToParameter(unittest.TestCase):

    @unittest.skipIf(om_version < (3, 4, 1) or (om_version == (3, 4, 1) and om_dev_version),
                     'test requires OpenMDAO >= 3.4.1')
    def test_connect_control_to_parameter(self):
        """ Test that the final value of a control in one phase can be connected as the value
        of a parameter in a subsequent phase. """
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.cannonball.size_comp import CannonballSizeComp

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
        ascent = dm.Phase(ode_class=CannonballODEVectorCD, transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)

        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, rate_source='r_dot', units='m')
        ascent.add_state('h', fix_initial=True, fix_final=False, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        ascent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        ascent.add_control('CD', targets=['CD'], opt=False, val=0.05)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial',
                                       upper=400000, lower=0, ref=100000)

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODEVectorCD, transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', units='m', rate_source='h_dot', fix_initial=False, fix_final=True)
        descent.add_state('gam', units='rad', rate_source='gam_dot', fix_initial=False, fix_final=False)
        descent.add_state('v', units='m/s', rate_source='v_dot', fix_initial=False, fix_final=False)

        descent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        descent.add_parameter('mass', targets=['m'], units='kg', static_target=True)
        descent.add_parameter('CD', targets=['CD'], val=0.01)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'mass', 'descent': 'mass'}, static_target=True)

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005, static_target=True)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        traj.connect('ascent.timeseries.controls:CD', 'descent.parameters:CD', src_indices=[-1])

        # A linear solver at the top level can improve performance.
        p.model.linear_solver = om.DirectSolver()

        # Finish Problem Setup
        p.setup()

        # Set Initial Guesses
        p.set_val('external_params.radius', 0.05, units='m')
        p.set_val('external_params.dens', 7.87, units='g/cm**3')

        p.set_val('traj.ascent.controls:CD', 0.5)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')

        dm.run_problem(p, simulate=True, make_plots=True)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1], 3183.25, tolerance=1.0E-2)
        assert_near_equal(p.get_val('traj.ascent.timeseries.controls:CD')[-1],
                          p.get_val('traj.descent.timeseries.parameters:CD')[0])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
