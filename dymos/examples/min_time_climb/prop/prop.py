import openmdao.api as om
from .mdot_comp import MassFlowRateComp
from .max_thrust_comp import MaxThrustComp
from .thrust_comp import ThrustComp


class PropGroup(om.Group):
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
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']

        max_thrust_comp = MaxThrustComp(vec_size=nn, extrapolate=True, method='cubic')
        # max_thrust_comp = BrysonMaxThrustComp(num_nodes=nn)

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
