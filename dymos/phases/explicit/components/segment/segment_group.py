from __future__ import print_function, division, absolute_import

from openmdao.api import Group

from .stage_state_comp import StageStateComp
from .stepsize_comp import StepSizeComp


class ExplicitSegment(Group):

    def initialize(self):

        self.options.declare('num_steps', types=(int,),
                             desc='the number of steps taken in the segment')
        self.options.declare('ode_class', types=(int,),
                             desc='the number of steps taken in the segment')
        self.options.declare('ode_init_kwargs', types=(int,),
                             desc='the number of steps taken in the segment')
        self.options.declare('state_options', types=dict)
        self.options.declare('method', types=str, default='rk4')

    def setup(self):
        num_steps = self.options['num_steps']
        state_options = self.options['state_options']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        num_stages = 4

        self.add_subsystem('h_comp', subsys=StepSizeComp(num_steps), promotes_inputs=['t_steps'])
        self.add_subsystem('stage_state_comp', subsys=StageStateComp(num_steps=num_steps,
                                                                     state_options=state_options))

        self.add_subsystem('stage_ode', subsys=ode_class(num_nodes=num_steps*num_stages, **ode_init_kwargs))
        self.add_subsystem('stage_k_comp', subsys=StageKComp(num_steps))
        self.add_subsystem('advance_comp', subsys=AdvanceComp())