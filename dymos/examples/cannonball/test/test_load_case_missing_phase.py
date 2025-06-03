import unittest

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal

import numpy as np
import openmdao.api as om

import dymos as dm
from dymos.examples.cannonball.size_comp import CannonballSizeComp
from dymos.examples.cannonball.cannonball_ode import CannonballODE


@use_tempdirs
class TestTwoPhaseCannonballLoadCase(unittest.TestCase):

    def _make_problem(self, ascent_only=False):
        """ Run ascent phase alone to create a restart file in which it is present. """
        p = om.Problem()

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['print_results'] = False
        p.driver.declare_coloring()

        p.model.add_subsystem('size_comp', CannonballSizeComp(), promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, units='m', rate_source='r_dot')
        ascent.add_state('h', fix_initial=True, fix_final=False,  units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        ascent.add_parameter('mass', targets=['m'], units='kg', static_target=True)
        ascent.add_parameter('CD', targets=['CD'], units=None, static_target=True)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000, shape=(1,))

        # Temporary objective while building the problem
        ascent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           val=0.5, units=None, opt=False)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        if ascent_only:
            traj.add_parameter('m', units='kg', val=1.0,
                               targets={'ascent': 'mass'})
        else:
            traj.add_parameter('m', units='kg', val=1.0,
                               targets={'ascent': 'mass', 'descent': 'mass'})

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005)

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        if ascent_only:
            return p

        # Second Phase (descent)
        # transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)
        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', fix_initial=False, fix_final=True,  units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        descent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        # Link Phases (link time and all state variables)
        # Note velocity is not included here.  Doing so is equivalent to linking kinetic energy,
        # and causes a duplicate row in the constraint jacobian.
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'h', 'gam'], connected=False)

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        p.setup()

        return p

    @require_pyoptsparse(optimizer='SLSQP')
    def test_load_case_missing_phase(self):
        """ Test that loading a case with a missing phase gracefully skips that phase. """

        p_ascent_only = self._make_problem(ascent_only=True)

        p_ascent_only.setup()

        traj = p_ascent_only.model.traj
        ascent = traj.phases.ascent

        # Set pre-trajectory values
        p_ascent_only.set_val('radius', 0.05, units='m')
        p_ascent_only.set_val('dens', 7.87, units='g/cm**3')

        # Trajectory parameters
        traj.set_parameter_val('CD', 0.5)

        # Phase ascent times
        ascent.set_time_val(initial=0.0, duration=10.0)

        # Phase ascent state guesses
        ascent.set_state_val('r', [0, 100])
        ascent.set_state_val('h', [0, 100])
        ascent.set_state_val('v', [200, 150])
        ascent.set_state_val('gam', [25, 0], units='deg')

        dm.run_problem(p_ascent_only)

        p = self._make_problem()

        p.setup()

        ascent = p.model.traj.phases.ascent
        descent = p.model.traj.phases.descent

        # Set pre-trajectory values
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        # Trajectory parameters
        traj.set_parameter_val('CD', 0.5)

        ascent.set_time_val(initial=0.0, duration=10.0)

        # Phase ascent state guesses
        ascent.set_state_val('r', [0, 100])
        ascent.set_state_val('h', [0, 100])
        ascent.set_state_val('v', [200, 150])
        ascent.set_state_val('gam', [25, 0], units='deg')

        # Phase descent times
        descent.set_time_val(initial=10.0, duration=10.0)

        # Phase descent state guesses
        descent.set_state_val('r', [100, 200])
        descent.set_state_val('h', [100, 0])
        descent.set_state_val('v', [150, 200])
        descent.set_state_val('gam', [0, -45], units='deg')

        sol_db = p_ascent_only.get_outputs_dir() / 'dymos_solution.db'

        case = om.CaseReader(sol_db).get_case('final')

        p.load_case(case)

        p.run_model()

        h_expected = np.array([0.00000000, 228.08533260, 434.90000516, 491.00415346,
                               592.60166806, 701.19757708, 730.03431228, 786.93071398,
                               848.54403903, 864.60323344, 895.67083544, 926.90913594,
                               934.25949414, 946.77982935, 954.88116543, 955.37021704])

        assert_near_equal(p.get_val('traj.ascent.states:h').ravel(), h_expected, tolerance=1.0E-5)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
