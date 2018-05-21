import numpy as np

from openmdao.api import ExplicitComponent


class CostComp(ExplicitComponent):
    """ Compute the operating cost of the flight as a combination of flight time and fuel."""
    def setup(self):
        self.add_input('tof', val=3600.0, desc='time of flight', units='s')
        self.add_input('initial_mass_fuel', val=20000.0, desc='initial aircraft fuel mass',
                       units='kg')

        self.add_output('cost', val=1.0, desc='operating cost', units='kg')

        self.declare_partials(of='cost', wrt='tof', val=5000.0 / 3600.0)
        self.declare_partials(of='cost', wrt='initial_mass_fuel', val=10.0)

    def compute(self, inputs, outputs):
        outputs['cost'] = 10.0 * inputs['initial_mass_fuel'] + 5000 * inputs['tof'] / 3600.0
