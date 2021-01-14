import openmdao.api as om
from ...models.atmosphere import USatm1976Comp
from .k_comp import KComp
from .aero_forces_comp import AeroForcesComp
from .lift_coef_comp import LiftCoefComp
from .ground_roll_eom_2d import GroundRollEOM2D
from .stall_speed_comp import StallSpeedComp


class GroundRollODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['h'],
                           promotes_outputs=['rho'])

        self.add_subsystem(name='k_comp',
                           subsys=KComp(num_nodes=nn),
                           promotes_inputs=['AR', 'span', 'e', 'h', 'h_w'],
                           promotes_outputs=['K'])

        self.add_subsystem(name='lift_coef_comp',
                           subsys=LiftCoefComp(num_nodes=nn),
                           promotes_inputs=['alpha', 'alpha_max', 'CL0', 'CL_max'],
                           promotes_outputs=['CL'])

        self.add_subsystem(name='aero_force_comp',
                           subsys=AeroForcesComp(num_nodes=nn),
                           promotes_inputs=['rho', 'v', 'S', 'CL', 'CD0', 'K'],
                           promotes_outputs=['q', 'L', 'D'])

        # Note: Typically a propulsion subsystem would go here, and provide thrust and mass
        # flow rate of the aircraft (for integrating mass).
        # In this simple demonstration, we're assuming the thrust of the aircraft is constant
        # and that the aircraft mass doesn't change (fuel burn during takeoff is negligible).

        self.add_subsystem(name='dynamics',
                           subsys=GroundRollEOM2D(num_nodes=nn),
                           promotes_inputs=['m', 'L', 'D', 'T', 'v', 'alpha', 'mu_r'],
                           promotes_outputs=['F_r', 'W', 'v_dot', 'r_dot'])

        self.add_subsystem(name='stall_speed_comp',
                           subsys=StallSpeedComp(num_nodes=nn),
                           promotes_inputs=['v', 'W', 'rho', 'CL_max', 'S'],
                           promotes_outputs=['v_stall', 'v_over_v_stall'])

        self.set_input_defaults('CL_max', val=2.0)
        self.set_input_defaults('alpha_max', val=10.0, units='deg')
