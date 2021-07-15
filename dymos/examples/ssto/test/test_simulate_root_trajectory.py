import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestSSTOSimulateRootTrajectory(unittest.TestCase):

    def test_ssto_simulate_root_trajectory(self):
        """
        Tests that we can properly simulate a trajectory even if the trajectory is the root
        group of the model.  In this case the name of the trajectory in the output will
        be 'sim_traj'.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm

        g = 1.61544  # lunar gravity, m/s**2

        class LaunchVehicle2DEOM(om.ExplicitComponent):
            """
            Simple 2D Cartesian Equations of Motion for a launch vehicle subject to thrust and drag.
            """
            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                # Inputs
                self.add_input('vx',
                               val=np.zeros(nn),
                               desc='x velocity',
                               units='m/s')

                self.add_input('vy',
                               val=np.zeros(nn),
                               desc='y velocity',
                               units='m/s')

                self.add_input('m',
                               val=np.zeros(nn),
                               desc='mass',
                               units='kg')

                self.add_input('theta',
                               val=np.zeros(nn),
                               desc='pitch angle',
                               units='rad')

                self.add_input('thrust',
                               val=2100000 * np.ones(nn),
                               desc='thrust',
                               units='N')

                self.add_input('Isp',
                               val=265.2 * np.ones(nn),
                               desc='specific impulse',
                               units='s')

                # Outputs
                self.add_output('xdot',
                                val=np.zeros(nn),
                                desc='velocity component in x',
                                units='m/s')

                self.add_output('ydot',
                                val=np.zeros(nn),
                                desc='velocity component in y',
                                units='m/s')

                self.add_output('vxdot',
                                val=np.zeros(nn),
                                desc='x acceleration magnitude',
                                units='m/s**2')

                self.add_output('vydot',
                                val=np.zeros(nn),
                                desc='y acceleration magnitude',
                                units='m/s**2')

                self.add_output('mdot',
                                val=np.zeros(nn),
                                desc='mass rate of change',
                                units='kg/s')

                # Setup partials
                ar = np.arange(self.options['num_nodes'])

                self.declare_partials(of='xdot', wrt='vx', rows=ar, cols=ar, val=1.0)
                self.declare_partials(of='ydot', wrt='vy', rows=ar, cols=ar, val=1.0)

                self.declare_partials(of='vxdot', wrt='vx', rows=ar, cols=ar)
                self.declare_partials(of='vxdot', wrt='m', rows=ar, cols=ar)
                self.declare_partials(of='vxdot', wrt='theta', rows=ar, cols=ar)
                self.declare_partials(of='vxdot', wrt='thrust', rows=ar, cols=ar)

                self.declare_partials(of='vydot', wrt='m', rows=ar, cols=ar)
                self.declare_partials(of='vydot', wrt='theta', rows=ar, cols=ar)
                self.declare_partials(of='vydot', wrt='vy', rows=ar, cols=ar)
                self.declare_partials(of='vydot', wrt='thrust', rows=ar, cols=ar)

                self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
                self.declare_partials(of='mdot', wrt='Isp', rows=ar, cols=ar)

            def compute(self, inputs, outputs):
                theta = inputs['theta']
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                vx = inputs['vx']
                vy = inputs['vy']
                m = inputs['m']
                F_T = inputs['thrust']
                Isp = inputs['Isp']

                outputs['xdot'] = vx
                outputs['ydot'] = vy
                outputs['vxdot'] = F_T * cos_theta / m
                outputs['vydot'] = F_T * sin_theta / m - g
                outputs['mdot'] = -F_T / (g * Isp)

            def compute_partials(self, inputs, jacobian):
                theta = inputs['theta']
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                m = inputs['m']
                F_T = inputs['thrust']
                Isp = inputs['Isp']

                # jacobian['vxdot', 'vx'] = -CDA * rho * vx / m
                jacobian['vxdot', 'm'] = -(F_T * cos_theta) / m ** 2
                jacobian['vxdot', 'theta'] = -(F_T / m) * sin_theta
                jacobian['vxdot', 'thrust'] = cos_theta / m

                # jacobian['vydot', 'vy'] = -CDA * rho * vy / m
                jacobian['vydot', 'm'] = -(F_T * sin_theta) / m ** 2
                jacobian['vydot', 'theta'] = (F_T / m) * cos_theta
                jacobian['vydot', 'thrust'] = sin_theta / m

                jacobian['mdot', 'thrust'] = -1.0 / (g * Isp)
                jacobian['mdot', 'Isp'] = F_T / (g * Isp ** 2)

        class LaunchVehicleLinearTangentODE(om.Group):
            """
            The LaunchVehicleLinearTangentODE for this case consists of a guidance component and
            the EOM.  Guidance is simply an OpenMDAO ExecComp which computes the arctangent of the
            tan_theta variable.
            """

            def initialize(self):
                self.options.declare('num_nodes', types=int,
                                     desc='Number of nodes to be evaluated in the RHS')

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem('guidance', om.ExecComp('theta=arctan(tan_theta)',
                                                           theta={'value': np.ones(nn),
                                                                  'units': 'rad'},
                                                           tan_theta={'value': np.ones(nn)}))

                self.add_subsystem('eom', LaunchVehicle2DEOM(num_nodes=nn))

                self.connect('guidance.theta', 'eom.theta')

        #
        # Setup and solve the optimal control problem
        #
        traj = dm.Trajectory()

        p = om.Problem(model=traj)

        phase = dm.Phase(ode_class=LaunchVehicleLinearTangentODE,
                         transcription=dm.Radau(num_segments=20, order=3, compressed=False))
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(10, 1000), units='s')

        #
        # Set the state options.  We include rate_source, units, and targets here since the ODE
        # is not decorated with their default values.
        #
        phase.add_state('x', fix_initial=True, lower=0, rate_source='eom.xdot')
        phase.add_state('y', fix_initial=True, lower=0, rate_source='eom.ydot')
        phase.add_state('vx', fix_initial=True, lower=0, rate_source='eom.vxdot',
                        units='m/s', targets=['eom.vx'])
        phase.add_state('vy', fix_initial=True, rate_source='eom.vydot',
                        units='m/s', targets=['eom.vy'])
        phase.add_state('m', fix_initial=True, rate_source='eom.mdot',
                        units='kg', targets=['eom.m'])

        #
        # The tangent of theta is modeled as a linear polynomial over the duration of the phase.
        #
        phase.add_polynomial_control('tan_theta', order=1, units=None, opt=True,
                                     targets=['guidance.tan_theta'])

        #
        # Parameters values for thrust and specific impulse are design parameters. They are
        # provided by an IndepVarComp in the phase, but with opt=False their values are not
        # design variables in the optimization problem.
        #
        phase.add_parameter('thrust', units='N', opt=False, val=3.0 * 50000.0 * 1.61544,
                            targets=['eom.thrust'])
        phase.add_parameter('Isp', units='s', opt=False, val=1.0E6, targets=['eom.Isp'])

        #
        # Set the boundary constraints.  These are all states which could also be handled
        # by setting fix_final=True and including the correct final value in the initial guess.
        #
        phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
        phase.add_boundary_constraint('vx', loc='final', equals=1627.0)
        phase.add_boundary_constraint('vy', loc='final', equals=0)

        phase.add_objective('time', index=-1, scaler=0.01)

        #
        # Add theta as a timeseries output since it's not included by default.
        #
        phase.add_timeseries_output('guidance.theta', units='deg')

        #
        # Set the optimizer
        #
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        #
        # We don't strictly need to define a linear solver here since our problem is entirely
        # feed-forward with no iterative loops.  It's good practice to add one, however, since
        # failing to do so can cause incorrect derivatives if iterative processes are ever
        # introduced to the system.
        #
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        #
        # Assign initial guesses for the independent variables in the problem.
        #
        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 500.0
        p['phase0.states:x'] = phase.interp('x', [0, 350000.0])
        p['phase0.states:y'] = phase.interp('y', [0, 185000.0])
        p['phase0.states:vx'] = phase.interp('vx', [0, 1627.0])
        p['phase0.states:vy'] = phase.interp('vy', [1.0E-6, 0])
        p['phase0.states:m'] = phase.interp('m', [50000, 50000])
        p['phase0.polynomial_controls:tan_theta'] = [[0.5 * np.pi], [0.0]]

        #
        # Solve the problem.
        #
        p.run_driver()

        #
        # Check the results.
        #
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 481, tolerance=0.01)

        #
        # Get the explitly simulated results
        #
        exp_out = traj.simulate()

        #
        # Plot the results
        #
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        axes[0].plot(p.get_val('phase0.timeseries.states:x'),
                     p.get_val('phase0.timeseries.states:y'),
                     marker='o',
                     ms=4,
                     linestyle='None',
                     label='solution')

        axes[0].plot(exp_out.get_val('sim_traj.phase0.timeseries.states:x'),
                     exp_out.get_val('sim_traj.phase0.timeseries.states:y'),
                     marker=None,
                     linestyle='-',
                     label='simulation')

        axes[0].set_xlabel('range (m)')
        axes[0].set_ylabel('altitude (m)')
        axes[0].set_aspect('equal')

        axes[1].plot(p.get_val('phase0.timeseries.time'),
                     p.get_val('phase0.timeseries.theta'),
                     marker='o',
                     ms=4,
                     linestyle='None')

        axes[1].plot(exp_out.get_val('sim_traj.phase0.timeseries.time'),
                     exp_out.get_val('sim_traj.phase0.timeseries.theta'),
                     linestyle='-',
                     marker=None)

        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('theta (deg)')

        plt.suptitle('Single Stage to Orbit Solution Using Polynomial Controls')
        fig.legend(loc='lower center', ncol=2)

        plt.show()


if __name__ == "__main__":
    unittest.main()
