import numpy as np
import openmdao.api as om


class KComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
        self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
        self.add_input('span', val=35.7, desc='Wingspan', units='m')
        self.add_input('h', val=np.ones(nn), desc='altitude', units='m')
        self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
        self.add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)

        ar = np.arange(nn)

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=False, show_summary=False)
        self.declare_partials(of='K', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        h = inputs['h']
        h_w = inputs['h_w']
        span = inputs['span']
        AR = inputs['AR']
        e = inputs['e']
        b = span / 2.0

        K_nom = 1.0 / (np.pi * AR * e)
        outputs['K'] = K_nom * 33 * ((h_w - h) / b)**1.5 / (1.0 + 33 * ((h_w - h) / b)**1.5)
