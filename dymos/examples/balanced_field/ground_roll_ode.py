import openmdao.api as om
from ...models.atmosphere import USatm1976Comp
from ..min_time_climb.aero import AeroGroup
from ..min_time_climb.prop import PropGroup
from .ground_roll_eom_2d import GroundRollEOM2D


class GroundRollODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['h'])

        self.add_subsystem(name='aero',
                           subsys=AeroGroup(num_nodes=nn),
                           promotes_inputs=['v', 'alpha', 'S'])

        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')

        self.add_subsystem(name='prop',
                           subsys=PropGroup(num_nodes=nn),
                           promotes_inputs=['h', 'Isp', 'throttle'])

        self.connect('aero.mach', 'prop.mach')

        self.add_subsystem(name='dynamics',
                           subsys=GroundRollEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'v', 'alpha'],
                           promotes_outputs=['F_r', 'W'])

        self.connect('aero.f_drag', 'dynamics.D')
        self.connect('aero.f_lift', 'dynamics.L')
        self.connect('prop.thrust', 'dynamics.T')
