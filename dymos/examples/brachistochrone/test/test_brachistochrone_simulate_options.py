import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.misc import is_unspecified


class BrachistochroneODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')
        self.add_input('g', val=9.80665, desc='acceleration of gravity', units='m/s**2')
        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')
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
class TestBrachistochroneSimulate(unittest.TestCase):

    def test_simulate_options(self):
        import itertools
        import numpy as np
        import openmdao.api as om
        import dymos as dm
        from dymos.utils.misc import _unspecified

        for method, atol, first_step, times_per_seg in \
                itertools.product(('RK23', 'RK45'), (0.01, 0.001), (None, 0.01, .1), (_unspecified, 10, 20)):
            with self.subTest(msg=f'{method} {atol} {first_step} {times_per_seg}'):

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
                                 transcription=dm.GaussLobatto(num_segments=5, order=3))

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

                # Minimize final time.
                phase.add_objective('time', loc='final')

                # Set the simulate options
                phase.set_simulate_options(method=method, atol=atol,
                                           first_step=first_step, times_per_seg=times_per_seg)

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
                phase.set_state_val('v', [0, 9.9], units='m/s')
                phase.set_control_val('theta', [90, 90], units='deg')

                # Run the driver to simulate the problem
                dm.run_problem(p, run_driver=False, simulate=True)

                sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

                sim = om.CaseReader(sim_db).get_case('final')
                t = sim.get_val('traj.phase0.timeseries.time')

                opt_times_per_seg = phase.simulate_options['times_per_seg']
                num_segments = phase.options['transcription'].grid_data.num_segments

                expected_num_times = opt_times_per_seg * num_segments if is_unspecified(times_per_seg) \
                    else times_per_seg * num_segments

                self.assertEqual(expected_num_times, t.size)

                dm.run_problem(p, run_driver=False, simulate=True, simulate_kwargs={'times_per_seg': 7})

                sim_db = traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

                sim = om.CaseReader(sim_db).get_case('final')
                t = sim.get_val('traj.phase0.timeseries.time')
                self.assertEqual(7 * num_segments, t.size)
