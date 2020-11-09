import numpy as np

import openmdao.api as om


class MaxThrustComp(om.ExplicitComponent):
    """ Compute the maximum thrust given the current aircraft state and its
        maximum sea-level thrust with a simple pressure correction.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.pres_sl = 101325.0  # Pa

        self.add_input(name='pres', shape=(nn,), desc='atmospheric pressure', units='Pa')

        self.add_input(name='max_thrust_sl', shape=(1,), desc='maximum thrust at sea-level',
                       units='N')

        self.add_output(name='max_thrust', shape=(nn,), desc='maximum thrust at current altitude',
                        units='N')

        ar = np.arange(nn)
        self.declare_partials('max_thrust', 'pres', rows=ar, cols=ar)
        self.declare_partials('max_thrust', 'max_thrust_sl', dependent=True)

    def compute(self, inputs, outputs):
        outputs['max_thrust'] = inputs['max_thrust_sl'] * inputs['pres'] / self.pres_sl

    def compute_partials(self, inputs, partials):
        partials['max_thrust', 'pres'] = inputs['max_thrust_sl'] / self.pres_sl
        partials['max_thrust', 'max_thrust_sl'] = inputs['pres'] / self.pres_sl
