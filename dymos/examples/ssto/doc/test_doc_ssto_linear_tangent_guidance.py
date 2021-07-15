import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestDocSSTOLinearTangentGuidance(unittest.TestCase):

    @save_for_docs
    def test_doc_ssto_linear_tangent_guidance(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.plotting import plot_results

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

        class LinearTangentGuidanceComp(om.ExplicitComponent):
            """ Compute pitch angle from static controls governing linear expression for
                pitch angle tangent as function of time.
            """

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_input('a_ctrl',
                               val=np.zeros(nn),
                               desc='linear tangent slope',
                               units='1/s')

                self.add_input('b_ctrl',
                               val=np.zeros(nn),
                               desc='tangent of theta at t=0',
                               units=None)

                self.add_input('time_phase',
                               val=np.zeros(nn),
                               desc='time',
                               units='s')

                self.add_output('theta',
                                val=np.zeros(nn),
                                desc='pitch angle',
                                units='rad')

                # Setup partials
                arange = np.arange(self.options['num_nodes'])

                self.declare_partials(of='theta', wrt='a_ctrl', rows=arange, cols=arange, val=1.0)
                self.declare_partials(of='theta', wrt='b_ctrl', rows=arange, cols=arange, val=1.0)
                self.declare_partials(of='theta', wrt='time_phase', rows=arange, cols=arange, val=1.0)

            def compute(self, inputs, outputs):
                a = inputs['a_ctrl']
                b = inputs['b_ctrl']
                t = inputs['time_phase']
                outputs['theta'] = np.arctan(a * t + b)

            def compute_partials(self, inputs, jacobian):
                a = inputs['a_ctrl']
                b = inputs['b_ctrl']
                t = inputs['time_phase']

                x = a * t + b
                denom = x ** 2 + 1.0

                jacobian['theta', 'a_ctrl'] = t / denom
                jacobian['theta', 'b_ctrl'] = 1.0 / denom
                jacobian['theta', 'time_phase'] = a / denom

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
                self.add_subsystem('guidance', LinearTangentGuidanceComp(num_nodes=nn))
                self.add_subsystem('eom', LaunchVehicle2DEOM(num_nodes=nn))
                self.connect('guidance.theta', 'eom.theta')

        #
        # Setup and solve the optimal control problem
        #
        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)

        phase = dm.Phase(ode_class=LaunchVehicleLinearTangentODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=5, compressed=True))

        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(10, 1000),
                               targets=['guidance.time_phase'])

        phase.add_state('x', fix_initial=True, lower=0, rate_source='eom.xdot', units='m')
        phase.add_state('y', fix_initial=True, lower=0, rate_source='eom.ydot', units='m')
        phase.add_state('vx', fix_initial=True, lower=0, rate_source='eom.vxdot', targets=['eom.vx'], units='m/s')
        phase.add_state('vy', fix_initial=True, rate_source='eom.vydot', targets=['eom.vy'], units='m/s')
        phase.add_state('m', fix_initial=True, rate_source='eom.mdot', targets=['eom.m'], units='kg')

        phase.add_boundary_constraint('y', loc='final', equals=1.85E5, linear=True)
        phase.add_boundary_constraint('vx', loc='final', equals=1627.0)
        phase.add_boundary_constraint('vy', loc='final', equals=0)

        phase.add_parameter('a_ctrl', units='1/s', opt=True, targets=['guidance.a_ctrl'])
        phase.add_parameter('b_ctrl', units=None, opt=True, targets=['guidance.b_ctrl'])
        phase.add_parameter('thrust', units='N', opt=False, val=3.0 * 50000.0 * 1.61544, targets=['eom.thrust'])
        phase.add_parameter('Isp', units='s', opt=False, val=1.0E6, targets=['eom.Isp'])

        phase.add_objective('time', index=-1, scaler=0.01)

        p.model.linear_solver = om.DirectSolver()

        phase.add_timeseries_output('guidance.theta', units='deg')

        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 500.0
        p['traj.phase0.states:x'] = phase.interp('x', [0, 350000.0])
        p['traj.phase0.states:y'] = phase.interp('y', [0, 185000.0])
        p['traj.phase0.states:vx'] = phase.interp('vx', [0, 1627.0])
        p['traj.phase0.states:vy'] = phase.interp('vy', [1.0E-6, 0])
        p['traj.phase0.states:m'] = phase.interp('m', [50000, 50000])
        p['traj.phase0.parameters:a_ctrl'] = -0.01
        p['traj.phase0.parameters:b_ctrl'] = 3.0

        dm.run_problem(p)

        #
        # Check the results.
        #
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 481, tolerance=0.01)

        #
        # Get the explitly simulated results
        #
        exp_out = traj.simulate()

        #
        # Plot the results
        #
        plot_results([('traj.phase0.timeseries.states:x', 'traj.phase0.timeseries.states:y',
                       'range (m)', 'altitude (m)'),
                      ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.theta',
                       'range (m)', 'altitude (m)')],
                     title='Single Stage to Orbit Solution Using Linear Tangent Guidance',
                     p_sol=p, p_sim=exp_out)

        plt.show()


if __name__ == "__main__":
    unittest.main()
