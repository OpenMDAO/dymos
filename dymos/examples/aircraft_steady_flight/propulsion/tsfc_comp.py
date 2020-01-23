import numpy as np

import openmdao.api as om


class SFCComp(om.ExplicitComponent):
    """ Compute the specific fuel consumption based on the altitude
    and the sea-level specific fuel consumption.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.Ka = 1.5E-4 * 9.80665  # Altitude correction factor

        # Inputs
        self.add_input(name='tsfc_sl',
                       desc='sea-level specific fuel consumption',
                       units='1/s')

        self.add_input(name='alt',
                       shape=(nn,),
                       desc='altitude',
                       units='m')

        # Outputs
        self.add_output(name='tsfc',
                        val=np.zeros(nn),
                        desc='specific fuel consumption',
                        units='1/s')

        # Partials
        ar = np.arange(nn)
        self.declare_partials('tsfc', 'tsfc_sl', rows=ar, cols=np.zeros(nn), val=1.0)
        self.declare_partials('tsfc', 'alt', rows=ar, cols=ar, val=-1.0E-6 * self.Ka)

    def compute(self, inputs, outputs):
        outputs['tsfc'] = inputs['tsfc_sl'] - 1.0E-6 * self.Ka * inputs['alt']
