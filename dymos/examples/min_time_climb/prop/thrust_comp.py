import numpy as np

import openmdao.api as om


class ThrustComp(om.ExplicitComponent):
    """ Computes mass flow rate for the F4's 2 J79 engines at full throttle. """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('max_thrust', shape=(nn,), desc='maximum thrust', units='N')
        self.add_input('throttle', val=np.ones(nn), desc='throttle parameter', units=None)

        # Outputs
        self.add_output(name='thrust', val=np.zeros(nn),
                        desc='vehicle thrust at given throttle value',
                        units='N')

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='thrust', wrt='max_thrust', rows=ar, cols=ar)
        self.declare_partials(of='thrust', wrt='throttle', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['thrust'] = inputs['max_thrust'] * inputs['throttle']

    def compute_partials(self, inputs, partials):
        partials['thrust', 'max_thrust'] = inputs['throttle']
        partials['thrust', 'throttle'] = inputs['max_thrust']
