from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent

class RangeRateComp(ExplicitComponent):
    """
    Calculates range rate based on true airspeed and flight path angle.
    """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('TAS', val=np.ones(nn), desc='True airspeed', units='m/s')

        self.add_input('gam', val=np.zeros(nn), desc='Flight path angle', units='rad')

        self.add_output('dXdt:r', val=np.ones(nn), desc='Velocity along the ground (no wind)',
                        units='m/s')

        # Setup partials
        ar = np.arange(self.metadata['num_nodes'])
        self.declare_partials(of='*', wrt='*', dependent=False)
        self.declare_partials(of='dXdt:r', wrt='TAS', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:r', wrt='gam', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        TAS = inputs['TAS']
        gam = inputs['gam']
        outputs['dXdt:r'] = TAS*np.cos(gam)

    def compute_partials(self, inputs, partials):
        TAS = inputs['TAS']
        gam = inputs['gam']

        partials['dXdt:r', 'TAS'] = np.cos(gam)
        partials['dXdt:r', 'gam'] = -TAS * np.sin(gam)
