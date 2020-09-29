import numpy as np

import openmdao.api as om


class DynamicPressureCompFD(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('partial_coloring', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='rho', shape=(nn,), desc='atmospheric density', units='kg/m**3')
        self.add_input(name='v', shape=(nn,), desc='air-relative velocity', units='m/s')

        self.add_output(name='q', shape=(nn,), desc='dynamic pressure', units='N/m**2')

        self.declare_partials(of='q', wrt='rho', method='fd')
        self.declare_partials(of='q', wrt='v', method='fd')

        if self.options['partial_coloring']:
            self.declare_coloring(wrt=['*'], method='fd', tol=1.0E-6, num_full_jacs=2,
                                  show_summary=True, show_sparsity=True, min_improve_pct=10.)

    def compute(self, inputs, outputs):
        outputs['q'] = 0.5 * inputs['rho'] * inputs['v'] ** 2
