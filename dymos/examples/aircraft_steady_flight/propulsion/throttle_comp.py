import numpy as np

import openmdao.api as om


class ThrottleComp(om.ExplicitComponent):
    """ Compute 'tau' (throttle parameter) which is the ratio of the current thrust
        (as determined to provie flight equilibrium) with the maximum thrust given
        the current aircraft state.

    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='thrust', shape=(nn,), desc='current thrust', units='N')
        self.add_input(name='max_thrust', shape=(nn,), desc='maximum possible thrust', units='N')

        # Outputs
        self.add_output(name='tau', shape=(nn,), desc='throttle parameter', units=None)

        # Partials
        ar = np.arange(nn)
        self.declare_partials('tau', 'thrust', rows=ar, cols=ar)
        self.declare_partials('tau', 'max_thrust', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['tau'] = inputs['thrust'] / inputs['max_thrust']

    def compute_partials(self, inputs, partials):
        partials['tau', 'thrust'] = 1.0 / inputs['max_thrust']
        partials['tau', 'max_thrust'] = -inputs['thrust'] / inputs['max_thrust']**2
