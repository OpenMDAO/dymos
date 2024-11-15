import numpy as np
import openmdao.api as om


class OscillatorODE(om.ExplicitComponent):
    """
    A Dymos ODE for a damped harmonic oscillator.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('x', shape=(nn,), desc='displacement', units='m')
        self.add_input('v', shape=(nn,), desc='velocity', units='m/s')
        self.add_input('k', shape=(nn,), desc='spring constant', units='N/m')
        self.add_input('c', shape=(nn,), desc='damping coefficient', units='N*s/m')
        self.add_input('m', shape=(nn,), desc='mass', units='kg')

        # self.add_output('x_dot', val=np.zeros(nn), desc='rate of change of displacement', units='m/s')
        self.add_output('v_dot', val=np.zeros(nn), desc='rate of change of velocity', units='m/s**2')

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        v = inputs['v']
        k = inputs['k']
        c = inputs['c']
        m = inputs['m']

        f_spring = -k * x
        f_damper = -c * v

        outputs['v_dot'] = (f_spring + f_damper) / m


class OscillatorVectorODE(om.ExplicitComponent):
    """
    A Dymos ODE for a damped harmonic oscillator with vectorized states.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('static_params', types=bool)

    def setup(self):
        nn = self.options['num_nodes']
        static = self.options['static_params']

        # Inputs
        self.add_input('x', val=np.ones((nn, 2)))
        if static:
            self.add_input('A', val=np.ones((2, 2)))
        else:
            self.add_input('A', val=np.ones((nn, 2, 2)))

        # Output
        self.add_output('x_dot', val=np.zeros((nn, 2)))

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        A = inputs['A']
        x = inputs['x']

        static = self.options['static_params']
        ein_sequence = 'jk, ik->ij' if static else 'ijk, ik->ij'
        outputs['x_dot'] = np.einsum(ein_sequence, A, x)
