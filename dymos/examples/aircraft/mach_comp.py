import numpy as np

from openmdao.api import ExplicitComponent


class MachComp(ExplicitComponent):
    """ Compute Mach number based on true airspeed and the local speed of sound. """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('sos', val=np.zeros(nn), desc='atmospheric speed of sound', units='m/s')
        self.add_input('TAS', val=np.zeros(nn), desc='true airspeed', units='m/s')

        self.add_output('mach', val=np.zeros(nn), desc='Mach number', units=None)

        # Setup partials
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='mach', wrt='TAS', rows=ar, cols=ar)
        self.declare_partials(of='mach', wrt='sos', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['mach'] = inputs['TAS'] / inputs['sos']

    def compute_partials(self, inputs, partials):
        partials['mach', 'TAS'] = 1.0 / inputs['sos']
        partials['mach', 'sos'] = -inputs['TAS'] / inputs['sos']**2
