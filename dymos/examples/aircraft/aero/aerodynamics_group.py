from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from .aero_coef_comp import AeroCoefComp
from .aero_forces_comp import AeroForcesComp


class AerodynamicsGroup(Group):
    """
    The purpose of the Aerodynamics is to compute the lift and
    drag forces on the aircraft.
    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        n = self.metadata['num_nodes']

        self.add_subsystem(name='aero_coef_comp',
                           subsys=AeroCoefComp(num_nodes=n),
                           promotes_inputs=['mach', 'alpha', 'h', 'eta'],
                           promotes_outputs=['CL', 'CD', 'CM'])

        self.add_subsystem(name='aero_forces_comp',
                           subsys=AeroForcesComp(num_nodes=n),
                           promotes_inputs=['q', 'S', 'CL', 'CD'],
                           promotes_outputs=['L', 'D'])
