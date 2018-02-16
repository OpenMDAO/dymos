from __future__ import absolute_import

from openmdao.api import Group

from .mdot_comp import MassFlowRateComp
from .bryson_max_thrust_comp import BrysonMaxThrustComp
from .mbi_max_thrust_comp import MBIMaxThrustComp
from .max_thrust_comp import MaxThrustComp
from .thrust_comp import ThrustComp


class PropGroup(Group):
    """
    The purpose of the PropGroup is to compute the propulsive forces on the
    aircraft in the body frame.

    Parameters
    ----------
    mach : float
        Mach number (unitless)
    alt : float
        altitude (m)
    Isp : float
        specific impulse (s)
    throttle : float
        throttle value nominally between 0.0 and 1.0 (unitless)

    Unknowns
    --------
    thrust : float
        Vehicle thrust force (N)
    mdot : float
        Vehicle mass accumulation rate (kg/s)

    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int,
                              desc='Number of nodes to be evaluated in the RHS')
        self.metadata.declare('thrust_model', values=['bryson', 'mbi', 'metamodel'],
                              default='metamodel', desc='Type of thrust model to be used')

    def setup(self):
        nn = self.metadata['num_nodes']

        if self.metadata['thrust_model'] == 'bryson':
            max_thrust_comp = BrysonMaxThrustComp(num_nodes=nn)
        elif self.metadata['thrust_model'] == 'mbi':
            max_thrust_comp = MBIMaxThrustComp(num_nodes=nn)
        else:
            max_thrust_comp = MaxThrustComp(num_nodes=nn, extrapolate=True, method='cubic')

        self.add_subsystem(name='max_thrust_comp',
                           subsys=max_thrust_comp,
                           promotes_inputs=['mach', 'h'],
                           promotes_outputs=['max_thrust'])

        self.add_subsystem(name='thrust_comp',
                           subsys=ThrustComp(num_nodes=nn),
                           promotes_inputs=['max_thrust', 'throttle'],
                           promotes_outputs=['thrust'])

        self.add_subsystem(name='mdot_comp',
                           subsys=MassFlowRateComp(num_nodes=nn),
                           promotes_inputs=['thrust', 'Isp'],
                           promotes_outputs=['m_dot'])
