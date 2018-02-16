import numpy as np

from openmdao.api import ExplicitComponent


class CLComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        # Inputs
        self.add_input('CLa', shape=(nn,), desc='alpha lift coefficient', units=None)
        self.add_input('alpha', shape=(nn,), desc='angle of attck', units='rad')

        # Outputs
        self.add_output(name='CL', val=np.zeros(nn), desc='lift coefficient', units=None)

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='CL', wrt='CLa', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['CL'][:] = inputs['CLa']*inputs['alpha']

    def compute_partials(self, inputs, partials):
        partials['CL', 'CLa'] = inputs['alpha']
        partials['CL', 'alpha'] = inputs['CLa']
