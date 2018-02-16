from __future__ import print_function, division, absolute_import

from openmdoc import ODEFunction
from .brachistochone_eom_comp import BrachistochroneEOM


class BrachistochroneODE(ODEFunction):

    def __init__(self):
        super(BrachistochroneODE, self).__init__(system_class=BrachistochroneEOM)

        self.declare_time(units='s')

        self.declare_state('x', rate_source='xdot', units='m')
        self.declare_state('y', rate_source='ydot', units='m')
        self.declare_state('v', rate_source='vdot', targets=['v'], units='m/s')

        self.declare_parameter('theta', targets=['theta'])
        self.declare_parameter('g', units='m/s**2', targets=['g'])
