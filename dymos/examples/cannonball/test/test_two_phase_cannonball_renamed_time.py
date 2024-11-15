import unittest

import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.cannonball.size_comp import CannonballSizeComp
from dymos.examples.cannonball.cannonball_ode import CannonballODE


@use_tempdirs
class TestTwoPhaseCannonballRenamedTime(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_rename_time(self):
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

        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100),
                                duration_ref=100, units='s', name='t')

        ascent.set_state_options('r', rate_source='r_dot', fix_initial=True, fix_final=False)
        ascent.set_state_options('h', rate_source='h_dot', fix_initial=True, fix_final=False)
        ascent.set_state_options('gam', rate_source='gam_dot', fix_initial=False, fix_final=True)
        ascent.set_state_options('v', rate_source='v_dot', fix_initial=False, fix_final=False)

        ascent.add_parameter('S', units='m**2', static_target=True)
        ascent.add_parameter('m', units='kg', static_target=True)

        ascent.add_boundary_constraint('ke', loc='initial',
                                       upper=400000, lower=0, ref=100000)

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s', name='t')

        descent.add_state('r', rate_source='r_dot')
        descent.add_state('h', rate_source='h_dot', fix_initial=False, fix_final=True)
        descent.add_state('gam', rate_source='gam_dot', fix_initial=False, fix_final=False)
        descent.add_state('v', rate_source='v_dot', fix_initial=False, fix_final=False)

        descent.add_parameter('S', units='m**2', static_target=True)
        descent.add_parameter('m', units='kg', static_target=True)

        descent.add_objective('r', loc='final', scaler=-1.0)

        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False, static_target=True)
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'm', 'descent': 'm'}, static_target=True)
        traj.add_parameter('S', units='m**2', val=0.005, static_target=True)

        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        p.model.linear_solver = om.DirectSolver()

        p.setup()

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

        dm.run_problem(p, simulate=True)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
