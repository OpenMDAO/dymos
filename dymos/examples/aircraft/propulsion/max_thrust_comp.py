from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class MaxThrustComp(ExplicitComponent):
    """ Compute the maximum thrust given the current aircraft state and its
        maximum sea-level thrust with a simple pressure correction.
    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('max_thrust_sl', types=float,
                              desc='maximum thrust at sea-level', units='N')

    def setup(self):
        nn = self.metadata['num_nodes']

        self.pres_sl = 101325.0  # Pa
        self.K = self.metadata['max_thrust_sl'] / self.pres_sl

        self.add_input(name='pres', shape=(nn,), desc='atmospheric pressure', units='Pa')
        self.add_input(name='temp', shape=(nn,), desc='atmospheric temperature', units='K')

        self.add_output(name='max_thrust',shape=(self.num_nodes,),
                        desc='maximum thrust at current altitude', units='N')

        ar = np.arange(nn)
        self.declare_partials('max_thrust', 'pres', rows=ar, cols=ar, val=self.K)

    def compute(self, inputs, outputs):
        outputs['max_thrust'] = self.K * inputs['pres']
