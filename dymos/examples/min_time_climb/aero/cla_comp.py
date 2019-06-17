import numpy as np

import openmdao.api as om


class CLaComp(om.ExplicitComponent):
    """ Computes the alpha lift coefficient for induced drag

    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('mach', shape=(nn,), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='CLa', val=np.ones(nn), desc='alpha lift coefficient', units=None)

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='CLa', wrt='mach', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        M = inputs['mach']

        idx_low = np.where(M < 1.15)[0]
        idx_high = np.where(M >= 1.15)[0]

        c2_low = 1.0 / np.cosh((M[idx_low] - 1.0) / 0.06)**2
        c2_high = 1.0 / np.cosh(0.15 / 0.06)**2

        outputs['CLa'][idx_low] = 3.44 + c2_low
        outputs['CLa'][idx_high] = 3.44 + c2_high - 0.96 / 0.63 * (M[idx_high] - 1.15)

    def compute_partials(self, inputs, partials):
        M = inputs['mach']

        idx_low = np.where(M < 1.15)[0]
        idx_high = np.where(M >= 1.15)[0]

        k = 50.0 / 3.0
        tanh = np.tanh(k * (M[idx_low] - 1.0))
        sech2 = 1.0 - tanh**2

        partials['CLa', 'mach'][idx_low] = -2.0 * k * tanh * sech2
        partials['CLa', 'mach'][idx_high] = -32.0 / 21.0
