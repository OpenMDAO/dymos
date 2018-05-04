from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class SFCComp(ExplicitComponent):
    """ Compute the specific fuel consumption based on the altitude
    and the sea-level specific fuel consumption.
    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.Ka = 1.5E-4 * 9.80665

        # Inputs
        self.add_input(name='SFC_SL', desc='sea-level specific fuel consumption', units='1/s')
        self.add_input(name='h', desc='altitude', units='m')

        # Outputs
        self.add_output(name='SFC', val=np.zeros(nn), desc='specific fuel consumption', units='1/s')

        # Partials
        ar = np.arange(nn)
        self.declare_partials('SFC', 'SFC_SL', rows=ar, cols=ar, val=1.0)
        self.declare_partials('SFC', 'h', rows=ar, cols=ar, val=-1.0E-6 * self.Ka)

    def compute(self, inputs, outputs):
        outputs['SFC'] = inputs['SFC_SL'] - 1.0E-6 * self.Ka * inputs['h']
