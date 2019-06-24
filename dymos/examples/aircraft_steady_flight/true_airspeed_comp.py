import numpy as np

import openmdao.api as om


class TrueAirspeedComp(om.ExplicitComponent):
    """ Compute Mach number based on true airspeed and the local speed of sound. """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('sos', val=np.zeros(nn), desc='atmospheric speed of sound', units='m/s')
        self.add_input('mach', val=np.zeros(nn), desc='Mach number', units=None)
        self.add_output('TAS', val=np.zeros(nn), desc='true airspeed', units='m/s')

        # Setup partials
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='TAS', wrt='mach', rows=ar, cols=ar)
        self.declare_partials(of='TAS', wrt='sos', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['TAS'] = inputs['mach'] * inputs['sos']

    def compute_partials(self, inputs, partials):
        partials['TAS', 'mach'] = inputs['sos']
        partials['TAS', 'sos'] = inputs['mach']
