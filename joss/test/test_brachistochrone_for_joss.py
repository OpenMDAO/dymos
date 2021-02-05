import unittest
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal


@use_tempdirs
class TestBrachistochroneForJOSS(unittest.TestCase):

    def test_results(self):

        # Begin code for paper
        import numpy as np
        import openmdao.api as om
        import dymos as dm
        import matplotlib.pyplot as plt

        # First define a system which computes the equations of motion
        class BrachistochroneEOM(om.ExplicitComponent):
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                # Inputs
                self.add_input('v', val=np.zeros(nn), units='m/s', desc='velocity')
                self.add_input('theta', val=np.zeros(nn), units='rad', desc='angle of wire')
                self.add_output('xdot', val=np.zeros(nn), units='m/s', desc='x rate of change')
                self.add_output('ydot', val=np.zeros(nn), units='m/s', desc='y rate of change')
                self.add_output('vdot', val=np.zeros(nn), units='m/s**2', desc='v rate of change')

                # Ask OpenMDAO to compute the partial derivatives using complex-step
                # with a partial coloring algorithm for improved performance
                self.declare_partials(of='*', wrt='*', method='cs')
                self.declare_coloring(wrt='*', method='cs', show_summary=True)

            def compute(self, inputs, outputs):
                v, theta = inputs.values()
                outputs['vdot'] = 9.80665 * np.cos(theta)
                outputs['xdot'] = v * np.sin(theta)
                outputs['ydot'] = -v * np.cos(theta)

        p = om.Problem()

        # Define a Trajectory object
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        # Define a Dymos Phase object with GaussLobatto Transcription
        tx = dm.GaussLobatto(num_segments=10, order=3)
        phase = dm.Phase(ode_class=BrachistochroneEOM, transcription=tx)

        traj.add_phase(name='phase0', phase=phase)

        # Set the time options
        phase.set_time_options(fix_initial=True,
                               duration_bounds=(0.5, 10.0))
        # Set the state options
        phase.add_state('x', rate_source='xdot',
                        fix_initial=True, fix_final=True)
        phase.add_state('y', rate_source='ydot',
                        fix_initial=True, fix_final=True)
        phase.add_state('v', rate_source='vdot',
                        fix_initial=True, fix_final=False)
        # Define theta as a control.
        phase.add_control(name='theta', units='rad',
                          lower=0, upper=np.pi)
        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine total
        # derivative sparsity pattern.
        # This works in conjunction with partial derivative
        # coloring to give a large speedup
        p.driver.declare_coloring()

        # Setup the problem
        p.setup()

        # Now that the OpenMDAO problem is setup, we can guess the
        # values of time, states, and controls.
        p.set_val('traj.phase0.t_duration', 2.0)

        # States and controls here use a linearly interpolated
        # initial guess along the trajectory.
        p.set_val('traj.phase0.states:x',
                  phase.interpolate(ys=[0, 10], nodes='state_input'),
                  units='m')
        p.set_val('traj.phase0.states:y',
                  phase.interpolate(ys=[10, 5], nodes='state_input'),
                  units='m')
        p.set_val('traj.phase0.states:v',
                  phase.interpolate(ys=[0, 5], nodes='state_input'),
                  units='m/s')
        # constant initial guess for control
        p.set_val('traj.phase0.controls:theta', 90, units='deg')

        # Run the driver to solve the problem and generate default plots of
        # state and control values vs time
        dm.run_problem(p, make_plots=True, simulate=True)

        # Additional custom plot of y vs x to show the actual wire shape
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        x = p.get_val('traj.phase0.timeseries.states:x', units='m')
        y = p.get_val('traj.phase0.timeseries.states:y', units='m')
        ax.plot(x,y, marker='o')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        fig.savefig('brachistochone_yx.png', bbox_inches='tight')
        # end code for paper
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-4)
        assert_near_equal(x[-1], 10, tolerance=1.0E-4)
        assert_near_equal(y[-1], 5, tolerance=1.0E-4)
