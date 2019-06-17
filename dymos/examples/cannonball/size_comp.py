import numpy as np

import openmdao.api as om


class CannonballSizeComp(om.ExplicitComponent):
    """
    Compute the reference area and mass of a cannonball with a given radius and density.

    Notes
    -----
    This component is not vectorized with 'num_nodes' as is the usual way with Dymos, but is instead
    intended to compute a scalar mass and reference area from scalar radius and density inputs. This
    component does not reside in the ODE but instead its outputs are connected to the trajectory via
    input design parameters.
    """
    def setup(self):
        self.add_input(name='radius', val=1.0, desc='cannonball radius', units='m')
        self.add_input(name='dens', val=7870., desc='cannonball density', units='kg/m**3')

        self.add_output(name='mass', shape=(1,), desc='cannonball mass', units='kg')
        self.add_output(name='S', shape=(1,), desc='aerodynamic reference area', units='m**2')

        self.declare_partials(of='mass', wrt='dens')
        self.declare_partials(of='mass', wrt='radius')

        self.declare_partials(of='S', wrt='radius')

    def compute(self, inputs, outputs):
        radius = inputs['radius']
        dens = inputs['dens']

        outputs['mass'] = (4/3.) * dens * np.pi * radius ** 3
        outputs['S'] = np.pi * radius ** 2

    def compute_partials(self, inputs, partials):
        radius = inputs['radius']
        dens = inputs['dens']

        partials['mass', 'dens'] = (4/3.) * np.pi * radius ** 3
        partials['mass', 'radius'] = 4. * dens * np.pi * radius ** 2

        partials['S', 'radius'] = 2 * np.pi * radius
