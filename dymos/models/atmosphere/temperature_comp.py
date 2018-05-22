import numpy as np
from openmdao.api import ExplicitComponent

from .constants import h_lower, h_upper, matrix


class TemperatureComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        # Inputs
        self.add_input(name='h',
                       val=np.zeros(n),
                       desc='Altitude',
                       units='m')

        self.add_output(name='temp',
                        val=288.15 * np.ones(n),
                        desc='Temperature',
                        units='K')

        rhs = np.array([288.16 - (6.5e-3) * h_lower, 216.65, -6.5e-3, 0])

        self.coefs = np.linalg.solve(matrix, rhs)

        arange = np.arange(n)
        self.declare_partials('temp', 'h', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        """
        Temperature model extracted from linear portion and constant
        portion of the standard atmosphere.
        """
        h_m = inputs['h']

        a, b, c, d = self.coefs

        tropos = h_m <= h_lower
        strato = h_m > h_upper
        smooth = np.logical_and(~tropos, ~strato)

        outputs['temp'][:] = 0.0
        outputs['temp'][:] += tropos * (288.16 - 6.5e-3 * h_m)
        outputs['temp'][:] += strato * 216.65
        outputs['temp'][:] += smooth * (a * h_m ** 3 + b * h_m ** 2 + c * h_m + d)

    def compute_partials(self, inputs, partials):
        h_m = inputs['h']
        a, b, c, d = self.coefs

        tropos = h_m <= h_lower
        strato = h_m > h_upper
        smooth = np.logical_and(~tropos, ~strato)

        partials['temp', 'h'][:] = 0.0
        partials['temp', 'h'] -= 6.5E-3 * tropos
        partials['temp', 'h'] += smooth * (3 * a * h_m ** 2 + 2 * b * h_m + c)
