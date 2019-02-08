import numpy as np
from openmdao.api import ExplicitComponent

from .constants import h_lower, h_upper, h_trans, epsilon, matrix


class PressureComp(ExplicitComponent):
    """
    Compute the local pressure at the given altitude
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        tmp1 = 1.0 - 0.0065 * h_lower / 288.16
        tmp2 = np.exp(-9.81 * epsilon / (288 * 216.65))
        rhs = np.array([101325 * tmp1 ** 5.2561,
                        22632 * tmp2,
                        (-101325 * 5.2561 * (0.0065 / 288.16) * tmp1**4.2561),
                        (22632 * (-9.81 / (288 * 216.65)) * tmp2)])
        self.coefs = np.linalg.solve(matrix, rhs)

        self.add_input('h', shape=(n,), desc='Altitude', units='m')

        self.add_output('pres', val=101325. * np.ones(n), desc='Pressure', units='Pa')

        arange = np.arange(n)
        self.declare_partials('pres', 'h', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        h_m = inputs['h']

        tropos = h_m <= h_lower
        strato = h_m > h_upper
        smooth = np.logical_and(~tropos, ~strato)

        a, b, c, d = self.coefs

        outputs['pres'][:] = 0.0
        outputs['pres'] += tropos * (101325 * (1 - 0.0065 * h_m / 288.16)**5.2561)
        outputs['pres'] += strato * (22632 * np.exp(-9.80665 * (h_m - h_trans) / (288 * 216.65)))
        outputs['pres'] += smooth * (a * h_m**3 + b * h_m**2 + c * h_m + d)

    def compute_partials(self, inputs, partials):
        h_m = inputs['h']
        a, b, c, d = self.coefs

        tropos = h_m <= h_lower
        strato = h_m > h_upper
        smooth = np.logical_and(~tropos, ~strato)

        partials['pres', 'h'][:] = 0.0
        partials['pres', 'h'] += tropos * (-12.0132 * (1 - 0.0000225569 * h_m)**4.2561)
        partials['pres', 'h'] += strato * (22632 * (-9.80665 / (288 * 216.65)) *
                                           np.exp(9.80665 * 11000 / (288 * 216.65)) *
                                           np.exp(-9.80665 * h_m / (288 * 216.65)))
        partials['pres', 'h'] += smooth * (3 * a * h_m**2 + 2 * b * h_m + c)
