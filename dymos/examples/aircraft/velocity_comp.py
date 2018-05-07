import numpy as np

from openmdao.api import ExplicitComponent


class TrueAirspeedComp(ExplicitComponent):
    """ Compute true airspeed based on Mach number and the local speed of sound. """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('mach', val=np.zeros(nn), desc='Mach number', units=None)
        self.add_input('sos', val=np.zeros(nn), desc='atmospheric speed of sound', units='m/s')

        self.add_output('TAS', val=np.zeros(nn), desc='true airspeed', units='m/s')

        # Setup partials
        ar = np.arange(self.metadata['num_nodes'])

        self.declare_partials(of='TAS', wrt='mach', rows=ar, cols=ar)
        self.declare_partials(of='TAS', wrt='sos', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['TAS'] = inputs['mach'] * inputs['sos']

    def compute_partials(self, inputs, partials):
        partials['TAS', 'mach'] = inputs['sos']
        partials['TAS', 'sos'] = inputs['mach']
