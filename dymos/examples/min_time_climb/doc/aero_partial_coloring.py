import openmdao.api as om
from .dynamic_pressure_comp_partial_coloring import DynamicPressureCompFD
from ..aero.dynamic_pressure_comp import DynamicPressureComp
from ..aero.lift_drag_force_comp import LiftDragForceComp
from ..aero.cd0_comp import CD0Comp
from ..aero.kappa_comp import KappaComp
from ..aero.cla_comp import CLaComp
from ..aero.cl_comp import CLComp
from ..aero.cd_comp import CDComp
from ..aero.mach_comp import MachComp


class AeroGroup(om.Group):
    """
    The purpose of the AeroGroup is to compute the aerodynamic forces on the
    aircraft in the body frame.

    Parameters
    ----------
    v : float
        air-relative velocity (m/s)
    sos : float
        local speed of sound (m/s)
    rho : float
        atmospheric density (kg/m**3)
    alpha : float
        angle of attack (rad)
    S : float
        aerodynamic reference area (m**2)

    """
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        self.options.declare('fd', types=bool, default=False, desc='If True, use fd for partials for dynamic pressure')
        self.options.declare('partial_coloring', types=bool, default=False,
                             desc='If True and fd is True, color the approximated partials of dynamic pressure')

    def setup(self):
        nn = self.options['num_nodes']

        if self.options['fd']:
            q_comp = DynamicPressureCompFD(num_nodes=nn, partial_coloring=self.options['partial_coloring'])
        else:
            q_comp = DynamicPressureComp(num_nodes=nn)

        self.add_subsystem(name='mach_comp',
                           subsys=MachComp(num_nodes=nn),
                           promotes_inputs=['v', 'sos'],
                           promotes_outputs=['mach'])

        self.add_subsystem(name='cd0_comp',
                           subsys=CD0Comp(num_nodes=nn),
                           promotes_inputs=['mach'],
                           promotes_outputs=['CD0'])

        self.add_subsystem(name='kappa_comp',
                           subsys=KappaComp(num_nodes=nn),
                           promotes_inputs=['mach'],
                           promotes_outputs=['kappa'])

        self.add_subsystem(name='cla_comp',
                           subsys=CLaComp(num_nodes=nn),
                           promotes_inputs=['mach'],
                           promotes_outputs=['CLa'])

        self.add_subsystem(name='CL_comp',
                           subsys=CLComp(num_nodes=nn),
                           promotes_inputs=['alpha', 'CLa'],
                           promotes_outputs=['CL'])

        self.add_subsystem(name='CD_comp',
                           subsys=CDComp(num_nodes=nn),
                           promotes_inputs=['CD0', 'alpha', 'CLa', 'kappa'],
                           promotes_outputs=['CD'])

        self.add_subsystem(name='q_comp',
                           subsys=q_comp,
                           promotes_inputs=['rho', 'v'],
                           promotes_outputs=['q'])

        self.add_subsystem(name='lift_drag_force_comp',
                           subsys=LiftDragForceComp(num_nodes=nn),
                           promotes_inputs=['CL', 'CD', 'q', 'S'],
                           promotes_outputs=['f_lift', 'f_drag'])
