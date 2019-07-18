import numpy as np

import openmdao.api as om


class CD0Comp(om.ExplicitComponent):
    """ Computes the zero-lift drag coefficient
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('mach', shape=(nn,), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='CD0', val=np.zeros(nn), desc='zero-lift drag coefficient', units=None)

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='CD0', wrt='mach', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        M = inputs['mach']

        idx_low = np.where(M < 1.15)[0]
        idx_high = np.where(M >= 1.15)[0]

        outputs['CD0'][idx_low] = 0.013 + 0.0144 * (1.0 + np.tanh((M[idx_low] - 0.98) / 0.06))
        outputs['CD0'][idx_high] = 0.013 + \
            0.0144 * (1.0 + np.tanh(0.17 / 0.06)) - 0.011 * (M[idx_high] - 1.15)

    def compute_partials(self, inputs, partials):
        M = inputs['mach']

        idx_low = np.where(M < 1.15)[0]
        idx_high = np.where(M >= 1.15)[0]

        k = 50.0 / 3.0

        partials['CD0', 'mach'][idx_low] = 0.24 / (np.cosh(k * (M[idx_low] - 0.98))**2)
        partials['CD0', 'mach'][idx_high] = -0.011
