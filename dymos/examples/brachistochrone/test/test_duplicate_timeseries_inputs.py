import io
from contextlib import redirect_stdout
import warnings
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from ..brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestDuplicateTimeseriesInput(unittest.TestCase):

    def _make_problem(self, transcription):

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
                         transcription=transcription(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(1.0, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0.01, upper=np.pi-0.01, shape=(1,))

        phase.add_timeseries_output('*')

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        p.set_val('traj.phase0.t_initial', 0.0, units='s')
        p.set_val('traj.phase0.t_duration', 5.0, units='s')

        p.set_val('traj.phase0.states:x',
                  phase.interp('x', [0, 10]),
                  units='m')

        p.set_val('traj.phase0.states:y',
                  phase.interp('y', [10, 5]),
                  units='m')

        p.set_val('traj.phase0.states:v',
                  phase.interp('v', [0, 5]),
                  units='m/s')

        p.set_val('traj.phase0.controls:theta',
                  phase.interp('theta', [0.01, 90]),
                  units='deg')

        return p

    def test_duplicate_timeseries_input_gl(self):
        p = self._make_problem(dm.GaussLobatto)

        with io.StringIO() as buf, redirect_stdout(buf):
            p.run_driver()
            output = buf.getvalue()

        # First check that the input duplication does not exist
        warning_str = 'WARNING: The following components have multiple inputs connected to ' \
                      'the same source, which can introduce unnecessary data transfer overhead:'

        if warning_str in output:
            raise AssertionError('Multiple inputs connected to the same source exist')

        # Now check that the outputs are still the same.
        x_rate_vals = p.get_val('traj.phase0.timeseries.state_rates:x')
        xdot_vals = p.get_val('traj.phase0.timeseries.xdot')

        print(p.get_val('traj.phase0.timeseries.states:x').T)
        print(p.get_val('traj.phase0.timeseries.states:y').T)
        print(p.get_val('traj.phase0.timeseries.states:v').T)
        print(p.get_val('traj.phase0.timeseries.controls:theta').T)

        assert_near_equal(x_rate_vals, xdot_vals)

    def test_duplicate_timeseries_input_radau(self):
        p = self._make_problem(dm.Radau)

        with io.StringIO() as buf, redirect_stdout(buf):
            p.run_model()
            output = buf.getvalue()

        # First check that the input duplication does not exist
        warning_str = 'WARNING: The following components have multiple inputs connected to ' \
                      'the same source, which can introduce unnecessary data transfer overhead:'

        if warning_str in output:
            raise AssertionError('Multiple inputs connected to the same source exist')

        # Now check that the outputs are still the same.
        x_rate_vals = p.get_val('traj.phase0.timeseries.state_rates:x')
        xdot_vals = p.get_val('traj.phase0.timeseries.xdot')

        assert_near_equal(x_rate_vals, xdot_vals)
