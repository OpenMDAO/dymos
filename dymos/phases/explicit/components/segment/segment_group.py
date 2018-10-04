from __future__ import print_function, division, absolute_import

from openmdao.api import Group, NonlinearBlockGS, DirectSolver, NewtonSolver

from .stage_state_comp import StageStateComp
from .stage_time_comp import StageTimeComp
from .stage_k_comp import StageKComp
from .advance_comp import AdvanceComp

from dymos.phases.options import TimeOptionsDictionary
from dymos.utils.rk_methods import rk_methods


class ExplicitSegment(Group):

    def initialize(self):

        self.options.declare('num_steps', types=(int,),
                             desc='the number of steps taken in the segment')
        self.options.declare('ode_class', desc='The ODE System class')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('state_options', types=dict)
        self.options.declare('time_options', types=TimeOptionsDictionary)
        self.options.declare('control_options', types=dict, default={})
        self.options.declare('design_parameter_options', types=dict, default={})
        self.options.declare('input_parameter_options', types=dict, default={})
        self.options.declare('method', types=str, default='rk4')

    def setup(self):
        num_steps = self.options['num_steps']
        state_options = self.options['state_options']
        time_options = self.options['time_options']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        method = self.options['method']
        num_stages = rk_methods[method]['num_stages']

        self.add_subsystem('stage_time_comp',
                           subsys=StageTimeComp(num_steps=4, method=method,
                                                time_options=time_options),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem('stage_state_comp',
                           subsys=StageStateComp(num_steps=num_steps,
                                                 state_options=state_options,
                                                 method=method),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem('stage_ode', subsys=ode_class(num_nodes=num_steps*num_stages,
                                                         **ode_init_kwargs))

        self.add_subsystem('stage_k_comp',
                           subsys=StageKComp(num_steps=4,
                                             method=method,
                                             time_options=time_options,
                                             state_options=state_options),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem('advance_comp',
                           subsys=AdvanceComp(num_steps=4, method=method,
                                              state_options=state_options),
                           promotes_inputs=['*'],
                           promotes_outputs=['*']
                           )

        self.linear_solver = DirectSolver()
        self.nonlinear_solver = NonlinearBlockGS()
        # self.nonlinear_solver = NewtonSolver()

        self.nonlinear_solver.options['atol'] = 1e-14
        self.nonlinear_solver.options['rtol'] = 1e-14
        # self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['err_on_maxiter'] = True
        # self.nonlinear_solver.options['max_sub_solves'] = 10
        self.nonlinear_solver.options['maxiter'] = 150
        self.nonlinear_solver.options['iprint'] = 2