from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent

from dymos.phases.options import TimeOptionsDictionary
from dymos.utils.rk_methods import rk_methods


class StageTimeComp(ExplicitComponent):
    """
    Computes the values of the time to pass to the ODE for a given stage
    """
    def initialize(self):
        self.options.declare('time_options', types=TimeOptionsDictionary)
        self.options.declare('num_steps', types=(int,),
                             desc='Number of steps to take within the segment.')
        self.options.declare('method', default='rk4', values=('rk4',))

    def setup(self):
        num_steps = self.options['num_steps']
        time_options = self.options['time_options']
        method = self.options['method']
        num_stages = rk_methods[method]['num_stages']
        c = rk_methods[self.options['method']]['c']

        self.add_input(name='seg_t0_tf',
                       val=np.array([0.0, 1.0]),
                       desc='initial and final time in the segment',
                       units=time_options['units'])

        self.add_input(name='t_initial_phase',
                       val=0.0,
                       desc='initial time of the phase',
                       units=time_options['units'])

        self.add_output(name='h',
                        val=np.ones((num_steps,)),
                        desc='size of the steps within the segment',
                        units=time_options['units'])

        self.add_output(name='t_stage',
                        val=np.ones((num_steps, num_stages)),
                        desc='Times at each stage of each step.',
                        units=time_options['units'])

        self.add_output(name='t_phase_stage',
                        val=np.ones((num_steps, num_stages)),
                        desc='Phase elapsed time at each stage of each step.',
                        units=time_options['units'])

        self.add_output(name='t_step',
                        val=np.ones((num_steps + 1,)),
                        desc='Times at each step.',
                        units=time_options['units'])

        self.add_output(name='t_phase_step',
                        val=np.ones((num_steps + 1,)),
                        desc='Phase elapsed time at each step.',
                        units=time_options['units'])

        self.add_output(name='dt_dstau',
                        val=1.0,
                        desc='Ratio of segment time duration to segment Tau duration (2.0)',
                        units=time_options['units'])

        self.declare_partials(of='h',
                              wrt='seg_t0_tf',
                              val=np.repeat(np.array([[-1.0, 1.0]]) / num_steps, num_steps, axis=0))

        self.declare_partials(of='dt_dstau',
                              wrt='seg_t0_tf',
                              val=np.array([-0.5, 0.5]))

        v = np.zeros((num_steps * num_stages, 2))
        r = np.repeat(np.arange(num_steps)/num_steps, num_stages)
        v[:, 1] = np.tile(c / num_steps, num_steps) + r
        v[:, 0] = v[::-1, 1]

        self.declare_partials(of='t_stage',
                              wrt='seg_t0_tf',
                              val=v)

        self.declare_partials(of='t_phase_stage',
                              wrt='seg_t0_tf',
                              val=v)

        self.declare_partials(of='t_phase_stage',
                              wrt='t_initial_phase',
                              val=-1.0)

        v = np.zeros((num_steps + 1, 2))
        v[:, 1] = np.linspace(0, 1.0, num_steps + 1)
        v[:, 0] = v[::-1, 1]

        self.declare_partials(of='t_step',
                              wrt='seg_t0_tf',
                              val=v)

        self.declare_partials(of='t_phase_step',
                              wrt='t_initial_phase',
                              val=-1.0)

        self.declare_partials(of='t_phase_step',
                              wrt='seg_t0_tf',
                              val=v)

    def compute(self, inputs, outputs):
        num_steps = self.options['num_steps']
        c = rk_methods[self.options['method']]['c']
        seg_t0, seg_tf = inputs['seg_t0_tf']

        outputs['h'] = ((seg_tf - seg_t0) / num_steps) * np.ones(num_steps)

        t_step_ends = np.linspace(seg_t0, seg_tf, num_steps + 1)

        outputs['t_stage'][:, :] = t_step_ends[:-1, np.newaxis] + np.outer(outputs['h'], c)

        outputs['t_phase_stage'][:, :] = outputs['t_stage'][:, :] - inputs['t_initial_phase']

        outputs['t_step'][:] = t_step_ends

        outputs['t_phase_step'][:] = outputs['t_step'][:] - inputs['t_initial_phase']

        outputs['dt_dstau'] = 0.5 * (seg_tf - seg_t0)
