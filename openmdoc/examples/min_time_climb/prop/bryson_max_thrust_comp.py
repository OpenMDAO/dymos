import numpy as np

from openmdao.api import ExplicitComponent


class BrysonMaxThrustComp(ExplicitComponent):
    """ Computes thrust for the F4's 2 J79 engines at full throttle. """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

        # Coefficient matrix from Bryson
        self._Q = np.array([[30.21,    -0.668,   -6.877,  1.951,   -0.1512],
                            [-33.80,    3.347,    18.13, -5.865,    0.4757],
                            [100.80,   -77.56,    5.441,  2.864,   -0.3355],
                            [-78.99,   101.40,   -30.28,  3.236,   -0.1089],
                            [18.74,    -31.60,    12.04, -1.785,    0.09417]])

    def setup(self):
        nn = self.metadata['num_nodes']

        # Inputs
        self.add_input('h', shape=(nn,), desc='altitude', units='ft')
        self.add_input('mach', shape=(nn,), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='max_thrust', val=np.zeros(nn),
                        desc='maximum thrust produced by 2 J79 engines',
                        units='lbf')

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='max_thrust', wrt='h', rows=ar, cols=ar)
        self.declare_partials(of='max_thrust', wrt='mach', rows=ar, cols=ar)
        self.declare_partials(of='max_thrust', wrt='h', method='fd')
        self.declare_partials(of='max_thrust', wrt='mach', method='fd')

    def compute(self, inputs, outputs):
        hh = np.vander(inputs['h']/10000.0, 5, increasing=True)
        mm = np.vander(inputs['mach'], 5, increasing=True)

        # Note: interpolation takes altitude in tens of thousand of feet
        #       interpolation gives thrust in thousands of lbf

        # If we do the matrix multiplications in a vectorized sense then
        # we just want to pull off the diagonal, everything else is irrelevant.
        # multiplication is of shapes (n, 5), (5, 5), (5, n) -> (n, n)
        outputs['max_thrust'] = 1000 * np.diagonal(np.dot(mm, np.dot(self._Q, hh.T)))
