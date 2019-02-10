from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from .runge_kutta_state_predict_comp import RungeKuttaStatePredictComp
from .runge_kutta_k_comp import RungeKuttaKComp


class RungeKuttaSegment(Group):

    def initialize(self):
        self.options.declare('ode_class',
                             desc='System defining the ODE')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')

    def setup(self):
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']

        self.add_subsystem('state_predict', RungeKuttaStatePredictComp())

        self.add_subsystem('ode', subsys=ode_class(num_nodes=1, **ode_init_kwargs))

        self.add_subsystem('k_comp', RungeKuttaKComp())

        self.add_subsystem('state_advance', RKStateAdvanceComp())
