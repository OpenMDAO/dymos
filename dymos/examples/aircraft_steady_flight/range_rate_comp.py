import numpy as np
import openmdao.api as om


class RangeRateComp(om.ExplicitComponent):
    """
    Calculates range rate based on true airspeed and flight path angle.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('TAS', val=np.ones(nn), desc='True airspeed', units='m/s')

        self.add_input('gam', val=np.zeros(nn), desc='Flight path angle', units='rad')

        self.add_output('dXdt:range', val=np.ones(nn), desc='Velocity along the ground (no wind)',
                        units='m/s')

        # Setup partials
        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='*', wrt='*', dependent=False)
        self.declare_partials(of='dXdt:range', wrt='TAS', rows=ar, cols=ar)
        self.declare_partials(of='dXdt:range', wrt='gam', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        TAS = inputs['TAS']
        gam = inputs['gam']
        outputs['dXdt:range'] = TAS*np.cos(gam)

    def compute_partials(self, inputs, partials):
        TAS = inputs['TAS']
        gam = inputs['gam']

        partials['dXdt:range', 'TAS'] = np.cos(gam)
        partials['dXdt:range', 'gam'] = -TAS * np.sin(gam)
