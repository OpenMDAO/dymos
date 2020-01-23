import numpy as np

import openmdao.api as om


class ThrustComp(om.ExplicitComponent):
    """ Compute thrust from the thrust coefficient
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='CT',
                       shape=(nn,),
                       desc='thrust coefficient',
                       units=None)

        self.add_input(name='q',
                       shape=(nn,),
                       desc='dynamic pressure',
                       units='Pa')

        self.add_input(name='S',
                       shape=(nn,),
                       desc='reference area',
                       units='m**2')

        # Outputs
        self.add_output(name='thrust',
                        val=np.zeros(nn),
                        desc='thrust',
                        units='N')

        # Partials
        ar = np.arange(nn)
        self.declare_partials('thrust', 'CT', rows=ar, cols=ar)
        self.declare_partials('thrust', 'q', rows=ar, cols=ar)
        self.declare_partials('thrust', 'S', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['thrust'] = inputs['CT'] * inputs['q'] * inputs['S']

    def compute_partials(self, inputs, partials):
        partials['thrust', 'CT'] = inputs['q'] * inputs['S']
        partials['thrust', 'q'] = inputs['CT'] * inputs['S']
        partials['thrust', 'S'] = inputs['CT'] * inputs['q']
