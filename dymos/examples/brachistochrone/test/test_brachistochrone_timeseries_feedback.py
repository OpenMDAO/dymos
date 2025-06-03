import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal


class BrachistochroneODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
        self.add_input('g', val=9.80665, desc='acceleration of gravity', units='m/s**2')
        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')
        self.add_input('final_time', val=0.0, desc='expected final time', units='s')
        self.add_output('xdot', val=np.zeros(nn), desc='horizontal velocity', units='m/s')
        self.add_output('ydot', val=np.zeros(nn), desc='vertical velocity', units='m/s')
        self.add_output('vdot', val=np.zeros(nn), desc='acceleration mag.', units='m/s**2')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials(of='vdot', wrt='g', rows=arange, cols=np.zeros(nn, dtype=int))
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        jacobian['vdot', 'g'] = cos_theta
        jacobian['vdot', 'theta'] = -g * sin_theta

        jacobian['xdot', 'v'] = sin_theta
        jacobian['xdot', 'theta'] = v * cos_theta

        jacobian['ydot', 'v'] = -cos_theta
        jacobian['ydot', 'theta'] = v * sin_theta


@use_tempdirs
class TestBrachistochroneTimeseriesFeedback(unittest.TestCase):

    def test_timeseries_feedback(self):
        import numpy as np
        import openmdao.api as om
        import dymos as dm

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
                         transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        phase.add_parameter('final_time', units='s', static_target=True, opt=False)
        traj.add_parameter('final_time', units='s', static_target=True, opt=False)

        traj.connect(src_name='phase0.timeseries.time', tgt_name='parameters:final_time', src_indices=om.slicer[-1, ...])

        # The feedback connection introduced iterative behavior, so now nonlinear solver and linear solver are needed.
        traj.nonlinear_solver = om.NonlinearBlockGS()
        traj.linear_solver = om.DirectSolver()

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

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 5], units='m/s')

        phase.set_control_val('theta', [90, 90])
        phase.set_parameter_val('final_time', 0.0, units='s')

        # Run the driver to solve the problem
        dm.run_problem(p, run_driver=True, simulate=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol_case = om.CaseReader(sol_db).get_case('final')
        sim_case = om.CaseReader(sim_db).get_case('final')

        final_time_timeseries_sol = sol_case.get_val('traj.phase0.timeseries.time')[-1, ...]
        final_time_traj_param_sol = sol_case.get_val('traj.parameter_vals:final_time')[0, 0]
        final_time_timeseries_sim = sim_case.get_val('traj.phase0.timeseries.time')[-1, ...]
        final_time_traj_param_sim = sim_case.get_val('traj.parameter_vals:final_time')[0, 0]

        p.model.list_outputs(prom_name=True)

        assert_near_equal(final_time_traj_param_sol, final_time_timeseries_sol)
        assert_near_equal(final_time_traj_param_sim, final_time_timeseries_sol)
        assert_near_equal(final_time_timeseries_sim, final_time_timeseries_sol)
