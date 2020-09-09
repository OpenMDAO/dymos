import numpy as np
import openmdao.api as om


class RobotArmODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('arm_length', types=float, default=5.0)

    def setup(self):
        nn = self.options['num_nodes']
        L = self.options['arm_length']

        # inputs: 6 states and 3 controls
        self.add_input(name='x0', val=np.ones(nn), desc='distance the arm protrudes from the pivot', units='m')
        self.add_input(name='x1', val=np.ones(nn), desc='horizontal angle of the arm', units='rad')
        self.add_input(name='x2', val=np.ones(nn), desc='vertical angle of the arm', units='rad')
        self.add_input(name='x3', val=np.ones(nn), desc='rate of change of arm protrusion', units='m/s')
        self.add_input(name='x4', val=np.ones(nn), desc='horizontal angular velocity', units='rad/s')
        self.add_input(name='x5', val=np.ones(nn), desc='vertical angular velocity', units='rad/s')

        self.add_input(name='u0', val=np.ones(nn), desc='arm length control', units='m**2/s**2')
        self.add_input(name='u1', val=np.ones(nn), desc='horizontal angle control', units='m**3*rad/s**2')
        self.add_input(name='u2', val=np.ones(nn), desc='vertical angle control', units='m**3*rad/s**2')

        # outputs: 6 state rates
        self.add_output(name='x0_dot', val=np.ones(nn), desc='rate of change of arm protrusion', units='m/s')
        self.add_output(name='x1_dot', val=np.ones(nn), desc='horizontal angular velocity', units='rad/s')
        self.add_output(name='x2_dot', val=np.ones(nn), desc='vertical angular velocity', units='rad/s')
        self.add_output(name='x3_dot', val=np.ones(nn), desc='second derivative of arm protrusion', units='m/s**2')
        self.add_output(name='x4_dot', val=np.ones(nn), desc='horizontal angular acceleration', units='rad/s**2')
        self.add_output(name='x5_dot', val=np.ones(nn), desc='vertical angular acceleration', units='rad/s**2')

        # set up the partials
        r = c = np.arange(nn)

        self.declare_partials(of='x0_dot', wrt='x3', rows=r, cols=c, val=1.0)

        self.declare_partials(of='x1_dot', wrt='x4', rows=r, cols=c, val=1.0)

        self.declare_partials(of='x2_dot', wrt='x5', rows=r, cols=c, val=1.0)

        self.declare_partials(of='x3_dot', wrt='x3', rows=r, cols=c, val=1.0)
        self.declare_partials(of='x3_dot', wrt='u0', rows=r, cols=c, val=1.0/L)

        self.declare_partials(of='x4_dot', wrt='x0', rows=r, cols=c)
        self.declare_partials(of='x4_dot', wrt='x2', rows=r, cols=c)
        self.declare_partials(of='x4_dot', wrt='x4', rows=r, cols=c, val=1.0)
        self.declare_partials(of='x4_dot', wrt='u1', rows=r, cols=c)
        self.declare_partials(of='x4_dot', wrt='u2', rows=r, cols=c, val=0.0)

        self.declare_partials(of='x5_dot', wrt='x0', rows=r, cols=c)
        self.declare_partials(of='x5_dot', wrt='x5', rows=r, cols=c, val=1.0)
        self.declare_partials(of='x5_dot', wrt='u2', rows=r, cols=c)

    def compute(self, inputs, outputs):
        L = self.options['arm_length']

        x0 = inputs['x0']
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']
        x5 = inputs['x5']

        u0 = inputs['u0']
        u1 = inputs['u1']
        u2 = inputs['u2']

        outputs['x0_dot'] = x3
        outputs['x1_dot'] = x4
        outputs['x2_dot'] = x5
        outputs['x3_dot'] = u0/L
        outputs['x4_dot'] = 3*u1/(((L - x0)**3 + x0**3)*np.sin(x2)**2)
        outputs['x5_dot'] = 3*u2/((L - x0)**3 + x0**3)

    def compute_partials(self, inputs, partials):
        L = self.options['arm_length']

        x0 = inputs['x0']
        x2 = inputs['x2']

        u1 = inputs['u1']
        u2 = inputs['u2']

        partials['x4_dot', 'x0'] = 9*u1*(L - 2*x0)/(L*(L**2 - 3*L*x0 + 3*x0**2)**2*np.sin(x2)**2)
        partials['x4_dot', 'x2'] = -24*u1*np.cos(x2)/(((L - x0)**3 + x0**3)*(3*np.sin(x2) - np.sin(3*x2)))
        partials['x4_dot', 'u1'] = 3/(((L - x0)**3 + x0**3)*np.sin(x2)**2)

        partials['x5_dot', 'x0'] = 9*u2*(L - 2*x0)/(L*(L**2 - 3*L*x0 + 3*x0**2)**2)
        partials['x5_dot', 'u2'] = 3/((L - x0)**3 + x0**3)
