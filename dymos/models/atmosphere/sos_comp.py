import numpy as np

from openmdao.api import ExplicitComponent


class SpeedOfSoundComp(ExplicitComponent):
    """ Compute the Local Speed of Sound in the atmosphere."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='temp', val=np.zeros(nn), desc='Temperature', units='K')

        # Outputs
        self.add_output(name='sos', val=200*np.ones(nn), desc='local speed of sound', units='m/s')

        # Product of the gas constant times the ratio of specific heats
        gamma = 1.4    # Ratio of specific heads
        gas_c = 287.0  # Gas constant
        self.K = gamma * gas_c

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='sos', wrt='temp', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['sos'] = np.sqrt(self.K * inputs['temp'])

    def compute_partials(self, inputs, partials):
        sos = np.sqrt(self.K * inputs['temp'])
        partials['sos', 'temp'] = 0.5 * self.K / sos
