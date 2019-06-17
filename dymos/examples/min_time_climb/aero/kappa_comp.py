import numpy as np

import openmdao.api as om


class KappaComp(om.ExplicitComponent):
    r""" Computes the term kappa in the drag equation:

    .. math::

        C_D = C_{D0} + \kappa C_{L\alpha} \alpha^2

    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('mach', shape=(nn,), desc='Mach number', units=None)

        # Outputs
        self.add_output(name='kappa', val=np.zeros(nn), desc='induced drag coefficient', units=None)

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='kappa', wrt='mach', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        M = inputs['mach']

        idx_low = np.where(M < 1.15)[0]
        idx_high = np.where(M >= 1.15)[0]

        outputs['kappa'][idx_low] = 0.54 + 0.15 * (1.0 + np.tanh((M[idx_low] - 0.9) / 0.06))
        outputs['kappa'][idx_high] = 0.54 + 0.15 * (1.0 + np.tanh(0.25 / 0.06)) \
            + 0.14 * (M[idx_high] - 1.15)

    def compute_partials(self, inputs, partials):
        M = inputs['mach']

        idx_low = np.where(M < 1.15)[0]
        idx_high = np.where(M >= 1.15)[0]

        k = 50.0 / 3.0
        tanh = np.tanh(k * (M[idx_low] - 0.9))
        sech2 = 1.0 - tanh**2

        partials['kappa', 'mach'][idx_low] = 2.5 * sech2
        partials['kappa', 'mach'][idx_high] = 0.14
