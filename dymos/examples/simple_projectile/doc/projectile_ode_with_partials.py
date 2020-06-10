import numpy as np
import openmdao.api as om


class ProjectileODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', types=(float, int), default=9.80665, desc='gravitational acceleration (m/s**2)')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('vx', shape=(nn,), desc='horizontal velocity', units='m/s')
        self.add_input('vy', shape=(nn,), desc='vertical velocity', units='m/s')

        self.add_output('x_dot', val=np.zeros(nn), desc='rate of change of horizontal position', units='m/s')
        self.add_output('y_dot', val=np.zeros(nn), desc='rate of change of vertical position', units='m/s')
        self.add_output('vx_dot', val=np.zeros(nn), desc='rate of change of horizontal velocity', units='m/s**2')
        self.add_output('vy_dot', val=np.zeros(nn), desc='rate of change of vertical velocity', units='m/s**2')

        ar = np.arange(nn, dtype=int)
        self.declare_partials('x_dot', 'vx', rows=ar, cols=ar, val=1.0)
        self.declare_partials('y_dot', 'vy', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        outputs['x_dot'] = inputs['vx']
        outputs['y_dot'] = inputs['vy']
        outputs['vx_dot'] = 0.0
        outputs['vy_dot'] = -self.options['g']