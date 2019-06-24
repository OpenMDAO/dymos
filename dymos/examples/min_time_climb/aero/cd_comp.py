import numpy as np

import openmdao.api as om


class CDComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('CD0', shape=(nn,), desc='zero-lift drag coefficient', units=None)
        self.add_input('CLa', shape=(nn,), desc='alpha lift coefficient', units=None)
        self.add_input('kappa', shape=(nn,), desc='induced drag coefficient', units=None)
        self.add_input('alpha', shape=(nn,), desc='angle of attack', units='rad')

        # Outputs
        self.add_output(name='CD', val=np.zeros(nn), desc='drag coefficient', units=None)

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='CD', wrt='CD0', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='CD', wrt='CLa', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CD', wrt='kappa', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['CD'] = inputs['CD0'] + inputs['CLa'] * inputs['kappa'] * inputs['alpha']**2

    def compute_partials(self, inputs, partials):
        partials['CD', 'CLa'] = inputs['kappa'] * inputs['alpha']**2
        partials['CD', 'alpha'] = 2.0 * inputs['CLa'] * inputs['kappa'] * inputs['alpha']
        partials['CD', 'kappa'] = inputs['CLa'] * inputs['alpha']**2
