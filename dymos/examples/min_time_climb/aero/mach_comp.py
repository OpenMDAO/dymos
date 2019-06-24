import numpy as np

import openmdao.api as om


class MachComp(om.ExplicitComponent):
    """ Compute the Mach number based on vehicle airspeed and local speed
    of sound.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='v', shape=(nn,), desc='velocity magnitude', units='m/s')
        self.add_input('sos', shape=(nn,), desc='alpha lift coefficient', units='m/s')

        # Outputs
        self.add_output(name='mach', val=0.7*np.ones(nn), desc='Mach number', units=None)

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='mach', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='mach', wrt='sos', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['mach'][:] = inputs['v'] / inputs['sos']

    def compute_partials(self, inputs, partials):
        partials['mach', 'v'] = 1.0 / inputs['sos']
        partials['mach', 'sos'] = -inputs['v'] / inputs['sos'] ** 2
