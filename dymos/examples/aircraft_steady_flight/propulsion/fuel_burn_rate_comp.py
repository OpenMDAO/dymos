import numpy as np

import openmdao.api as om


class FuelBurnRateComp(om.ExplicitComponent):
    """ Computes the fuel burn rate (rate of change of fuel weight) based on SFC and thrust. """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input(name='thrust', shape=(nn,), desc='current thrust', units='N')
        self.add_input(name='tsfc', shape=(nn,), desc='specific fuel consumption', units='1/s')

        # Outputs
        self.add_output(name='dXdt:mass_fuel', shape=(nn,),
                        desc='rate of aircraft mass change - negative when fuel is being depleted',
                        units='kg/s')

        # Partials
        ar = np.arange(nn)
        self.declare_partials('dXdt:mass_fuel', 'thrust', rows=ar, cols=ar)
        self.declare_partials('dXdt:mass_fuel', 'tsfc', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['dXdt:mass_fuel'] = -inputs['tsfc'] * inputs['thrust'] / 9.80665

    def compute_partials(self, inputs, partials):
        partials['dXdt:mass_fuel', 'thrust'] = -inputs['tsfc'] / 9.80665
        partials['dXdt:mass_fuel', 'tsfc'] = -inputs['thrust'] / 9.80665
