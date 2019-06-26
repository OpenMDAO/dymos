import numpy as np

import openmdao.api as om


class MassComp(om.ExplicitComponent):
    """ Compute the total mass of the aircraft """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('mass_fuel', val=np.ones(nn), desc='fuel mass', units='kg')
        self.add_input('mass_empty', val=np.ones(nn), desc='empty aircraft mass', units='kg')
        self.add_input('mass_payload', val=np.ones(nn), desc='aircraft payload mass', units='kg')

        self.add_output('mass_total', val=np.ones(nn), desc='total aircraft mass', units='kg')
        self.add_output('W_total', val=np.ones(nn), desc='total aircraft weight', units='N')

        # Setup partials
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='mass_total', wrt='mass_fuel', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='mass_total', wrt='mass_empty', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='mass_total', wrt='mass_payload', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='W_total', wrt='mass_fuel', rows=ar, cols=ar, val=9.80665)
        self.declare_partials(of='W_total', wrt='mass_empty', rows=ar, cols=ar, val=9.80665)
        self.declare_partials(of='W_total', wrt='mass_payload', rows=ar, cols=ar, val=9.80665)

    def compute(self, inputs, outputs):
        outputs['mass_total'] = inputs['mass_fuel'] + inputs['mass_empty'] + inputs['mass_payload']
        outputs['W_total'] = 9.80665 * outputs['mass_total']
