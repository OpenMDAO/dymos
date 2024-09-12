import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.misc import om_version


@use_tempdirs
class TestBrachistochroneSimulate_units(unittest.TestCase):

    def test_brachistochrone_simulate_units(self):

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=BrachistochroneODE,
                         ode_init_kwargs={'static_gravity': True},
                         transcription=dm.GaussLobatto(num_segments=7, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, rate_source='ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, rate_source='vdot')

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi)

        phase.add_parameter(name='g', units='m/s**2', static_target=True, opt=False)

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        p.model.set_input_defaults("traj.phase0.parameters:g", val=9.80665/0.3048, units="ft/s**2")

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.
        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [1E-6, 9.9], units='m/s')
        phase.set_control_val('theta', [5, 100], units='deg')
        phase.set_parameter_val('g', 9.80665, units='m/s**2')

        # Run the driver to solve the problem
        dm.run_problem(p, simulate=True)

        sol_db = 'dymos_solution.db'
        sim_db = 'dymos_simulation.db'
        if om_version()[0] > (3, 34, 2):
            sol_db = p.get_outputs_dir() / sol_db
            sim_db = traj.sim_prob.get_outputs_dir() / sim_db

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')

        assert_near_equal(sim_case.get_val('traj.phase0.parameter_vals:g', units='m/s**2')[0],
                          sol_case.get_val('traj.phase0.parameter_vals:g', units='m/s**2')[0])

        assert_near_equal(sol_case.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-4)
        assert_near_equal(sim_case.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-4)
