import numpy as np
from openmdao.api import ExplicitComponent


class DensityComp(ExplicitComponent):
    """
    Density model using standard atmosphere model with troposphere,
    stratosphere.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        # Inputs
        self.add_input('pres',
                       val=101325. * np.ones(n),
                       desc='Pressure',
                       units='Pa')

        self.add_input('temp',
                       val=288.15 * np.ones(n),
                       desc='Temperature',
                       units='K')

        # Outputs
        self.add_output(name='rho',
                        val=1.225 * np.ones(n),
                        desc='Density',
                        units='kg/m**3')

        arange = np.arange(n)
        self.declare_partials('rho', '*', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        """ Density model extracted from the standard atmosphere. Depends on
        the temperature and the altitude. Model is valid for troposphere and
        stratosphere, and accounts for the linear decreasing temperature
        segment (troposphere), and the constant temperature segment.
        (stratosphere)
        """
        outputs['rho'] = inputs['pres'] / (288.0 * inputs['temp'])

    def compute_partials(self, inputs, partials):
        partials['rho', 'pres'] = 1.0 / (288.0 * inputs['temp'])
        partials['rho', 'temp'] = -inputs['pres'] / (288 * inputs['temp'] ** 2)
