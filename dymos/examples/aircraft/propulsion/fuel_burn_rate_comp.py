from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class FuelBurnRateComp(ExplicitComponent):
    """ Computes the fuel burn rate (rate of change of fuel weight) based on SFC and thrust. """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        # Inputs
        self.add_input(name='thrust', shape=(nn,), desc='current thrust', units='N')
        self.add_input(name='tsfc', shape=(nn,), desc='specific fuel consumption', units='1/s')

        # Outputs
        self.add_output(name='dXdt:W_f', shape=(nn,),
                        desc='rate of fuel flow - negative when fuel is being depleted', units=None)

        # Partials
        ar = np.arange(nn)
        self.declare_partials('dXdt:W_f', 'thrust', rows=ar, cols=ar)
        self.declare_partials('dXdt:W_f', 'tsfc', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['dXdt:W_f'] = -inputs['tsfc'] * inputs['thrust']

    def compute_partials(self, inputs, partials):
        partials['dXdt:W_f', 'thrust'] = -inputs['tsfc']
        partials['dXdt:W_f', 'tsfc'] = -inputs['thrust']
