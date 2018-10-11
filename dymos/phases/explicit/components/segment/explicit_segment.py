from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import Group, NonlinearBlockGS, DirectSolver

from .stage_state_comp import StageStateComp
from .stage_time_comp import StageTimeComp
from .stage_control_comp import StageControlComp
from .stage_k_comp import StageKComp
from .advance_comp import AdvanceComp
from ...solvers.nl_rk_solver import NonlinearRK

from dymos.phases.options import TimeOptionsDictionary
from dymos.phases.grid_data import GridData
from dymos.utils.rk_methods import rk_methods


class ExplicitSegment(Group):

    def initialize(self):

        self.options.declare('index', types=(int,), desc='the index of this segment in the phase')
        self.options.declare('num_steps', types=(int,),
                             desc='the number of steps taken in the segment')
        self.options.declare('ode_class', desc='The ODE System class')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('state_options', types=dict)
        self.options.declare('time_options', types=TimeOptionsDictionary)
        self.options.declare('control_options', types=dict, default={})
        self.options.declare('grid_data', types=GridData, allow_none=True, default=None)
        self.options.declare('design_parameter_options', types=dict, default={})
        self.options.declare('input_parameter_options', types=dict, default={})
        self.options.declare('method', types=str, default='rk4')

    def setup(self):
        idx = self.options['index']
        num_steps = self.options['num_steps']
        state_options = self.options['state_options']
        time_options = self.options['time_options']
        control_options = self.options['control_options']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        method = self.options['method']
        grid_data = self.options['grid_data']
        num_stages = rk_methods[method]['num_stages']

        self.add_subsystem('stage_time_comp',
                           subsys=StageTimeComp(num_steps=num_steps, method=method,
                                                time_options=time_options),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.connect('t_stage',
                     ['stage_ode.{0}'.format(t) for t in time_options['targets']],
                     src_indices=np.arange(num_steps * num_stages, dtype=int),
                     flat_src_indices=True)

        if control_options:
            ode_parameters = self.options['ode_class'].ode_options._parameters

            self.add_subsystem('stage_control_comp',
                               subsys=StageControlComp(index=idx,
                                                       num_steps=num_steps,
                                                       control_options=control_options,
                                                       method=method,
                                                       grid_data=grid_data),
                               promotes_inputs=['*'],
                               promotes_outputs=['*'])

            for control_name, options in iteritems(control_options):
                shape = options['shape']
                size = np.prod(shape)

                if control_name in ode_parameters:
                    val_src = 'stage_control_values:{0}'.format(control_name)
                    targets = ode_parameters[control_name]['targets']
                    self.connect(val_src,
                                 ['stage_ode.{0}'.format(t) for t in targets],
                                 src_indices=np.arange(num_steps * num_stages * size, dtype=int),
                                 flat_src_indices=True
                                 )

                if options['rate_param']:
                    rate_src = 'stage_control_rates:{0}_rate'.format(control_name)
                    targets = ode_parameters[options['rate_param']]['targets']

                    self.connect(rate_src,
                                 ['stage_ode.{0}'.format(t) for t in targets],
                                 src_indices=np.arange(num_steps * num_stages * size, dtype=int),
                                 flat_src_indices=True
                                 )

                if options['rate2_param']:
                    rate2_src = 'stage_control_rates:{0}_rate2'.format(control_name)
                    targets = ode_parameters[options['rate2_param']]['targets']

                    self.connect(rate2_src,
                                 ['stage_ode.{0}'.format(t) for t in targets],
                                 src_indices=np.arange(num_steps * num_stages * size, dtype=int),
                                 flat_src_indices=True
                                 )

        self.add_subsystem('stage_state_comp',
                           subsys=StageStateComp(num_steps=num_steps,
                                                 state_options=state_options,
                                                 method=method),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        for state_name, options in iteritems(state_options):
            size = np.prod(options['shape'])
            self.connect('stage_ode.{0}'.format(options['rate_source']),
                         'state_rates:{0}'.format(state_name),
                         src_indices=np.arange(num_steps * num_stages * size,
                                               dtype=int).reshape((num_steps, num_stages, size)),
                         flat_src_indices=True)

            self.connect('stage_states:{0}'.format(state_name),
                         ['stage_ode.{0}'.format(t) for t in options['targets']],
                         src_indices=np.arange(num_steps * num_stages * size, dtype=int),
                         flat_src_indices=True)

        self.add_subsystem('stage_ode', subsys=ode_class(num_nodes=num_steps*num_stages,
                                                         **ode_init_kwargs))

        self.add_subsystem('stage_k_comp',
                           subsys=StageKComp(num_steps=num_steps,
                                             method=method,
                                             time_options=time_options,
                                             state_options=state_options),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem('advance_comp',
                           subsys=AdvanceComp(num_steps=num_steps, method=method,
                                              state_options=state_options),
                           promotes_inputs=['*'],
                           promotes_outputs=['*']
                           )

        self.linear_solver = DirectSolver()
        self.nonlinear_solver = NonlinearBlockGS()
        self.nonlinear_solver = NonlinearRK()
        # from openmdao.api import NewtonSolver
        # self.nonlinear_solver = NewtonSolver()

        # self.nonlinear_solver.options['atol'] = 1e-14
        # self.nonlinear_solver.options['rtol'] = 1e-14
        # # self.nonlinear_solver.options['solve_subsystems'] = True
        # self.nonlinear_solver.options['err_on_maxiter'] = True
        # # self.nonlinear_solver.options['max_sub_solves'] = 10
        self.nonlinear_solver.options['maxiter'] = 150
        # self.nonlinear_solver.options['iprint'] = 2