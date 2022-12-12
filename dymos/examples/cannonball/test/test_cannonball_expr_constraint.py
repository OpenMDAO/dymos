import unittest
import matplotlib.pyplot as plt

from openmdao.utils.testing_utils import use_tempdirs
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.cannonball.size_comp import CannonballSizeComp
from dymos.examples.cannonball.cannonball_ode import CannonballODE


plt.switch_backend('Agg')


@use_tempdirs
class TestCannonballBoundaryConstraint(unittest.TestCase):

    def test_cannonball_design_boundary_constraint_expression(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
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

        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100),
                                duration_ref=100, units='s')

        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, rate_source='r_dot', units='m')
        ascent.add_state('h', fix_initial=True, fix_final=False, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', units='m**2', dynamic=False)
        ascent.add_parameter('m', units='kg', dynamic=False)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke = 0.5 * m * v**2', loc='initial',
                                       upper=400000, lower=0, ref=100000)

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', units='m', rate_source='h_dot', fix_initial=False, fix_final=True)
        descent.add_state('gam', units='rad', rate_source='gam_dot', fix_initial=False, fix_final=False)
        descent.add_state('v', units='m/s', rate_source='v_dot', fix_initial=False, fix_final=False)

        descent.add_parameter('S', units='m**2', dynamic=False)
        descent.add_parameter('m', units='kg', dynamic=False)

        descent.add_objective('r', loc='final', scaler=-1.0)

        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False, dynamic=False)

        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'm', 'descent': 'm'}, dynamic=False)

        traj.add_parameter('S', units='m**2', val=0.005, dynamic=False)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        p.model.linear_solver = om.DirectSolver()

        # Finish Problem Setup
        p.setup()

        # Set constants and initial guesses
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        p.set_val('traj.parameters:CD', 0.5)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interp(ys=[0, 100], nodes='state_input'))
        p.set_val('traj.ascent.states:h', ascent.interp(ys=[0, 100], nodes='state_input'))
        p.set_val('traj.ascent.states:v', ascent.interp(ys=[200, 150], nodes='state_input'))
        p.set_val('traj.ascent.states:gam', ascent.interp(ys=[25, 0], nodes='state_input'), units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp(ys=[100, 200], nodes='state_input'))
        p.set_val('traj.descent.states:h', descent.interp(ys=[100, 0], nodes='state_input'))
        p.set_val('traj.descent.states:v', descent.interp(ys=[150, 200], nodes='state_input'))
        p.set_val('traj.descent.states:gam', descent.interp(ys=[0, -45], nodes='state_input'), units='deg')

        # Run the optimization and final explicit simulation
        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)


if __name__ == '__main__':
    unittest.main()
