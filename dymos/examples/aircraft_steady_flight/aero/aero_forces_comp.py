import numpy as np

import openmdao.api as om


class AeroForcesComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('q', val=np.ones(nn), desc='dynamic pressure', units='Pa')
        self.add_input('S', val=np.ones(nn), desc='aerodynamic reference area', units='m**2')
        self.add_input('CL', val=np.ones(nn), desc='lift coefficient', units=None)
        self.add_input('CD', val=np.ones(nn), desc='drag coefficient', units=None)

        self.add_output('L', val=np.ones(nn), desc='lift', units='N')
        self.add_output('D', val=np.ones(nn), desc='drag', units='N')

        ar = np.arange(nn)

        self.declare_partials('L', 'q', rows=ar, cols=ar)
        self.declare_partials('L', 'S', rows=ar, cols=ar)
        self.declare_partials('L', 'CL', rows=ar, cols=ar)

        self.declare_partials('D', 'q', rows=ar, cols=ar)
        self.declare_partials('D', 'S', rows=ar, cols=ar)
        self.declare_partials('D', 'CD', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        qS = inputs['q'] * inputs['S']

        outputs['L'] = qS * inputs['CL']
        outputs['D'] = qS * inputs['CD']

    def compute_partials(self, inputs, partials):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']
        qS = inputs['q'] * inputs['S']

        partials['L', 'q'] = S * CL
        partials['L', 'S'] = q * CL
        partials['L', 'CL'] = qS

        partials['D', 'q'] = S * CD
        partials['D', 'S'] = q * CD
        partials['D', 'CD'] = qS
