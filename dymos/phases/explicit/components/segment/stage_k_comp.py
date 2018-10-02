from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import ExplicitComponent, OptionsDictionary

from dymos.utils.misc import get_rate_units


class StageKComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('time_options', types=OptionsDictionary)
        self.options.declare('state_options', types=dict)
        self.options.declare('num_steps', types=int)
        self.options.declare('method', default='rk4', values=('rk4',))

        self.var_names = {}

    def setup(self):
        time_units = self.options['time_options']['units']
        num_steps = self.options['num_steps']
        num_stages = 4

        self.add_input('h', val=np.ones(num_steps), units=time_units,
                       desc='step size for the RK method')

        for state_name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.var_names[state_name] = {}
            self.var_names[state_name]['f'] = 'state_rates:{0}'.format(state_name)
            self.var_names[state_name]['k'] = 'k:{0}'.format(state_name)

            self.add_input(
                name=self.var_names[state_name]['f'],
                shape=tuple([num_steps, num_stages] + list(shape)),
                desc='Rate of state {0} for each step/stage in the segment.'.format(state_name),
                units=rate_units)

            self.add_output(
                name=self.var_names[state_name]['k'],
                shape=tuple([num_steps, num_stages] + list(shape)),
                desc='RK multiplier k {0} for each step/stage in the segment.'.format(state_name),
                units=units)


            ar = np.arange(num_stages * num_steps, dtype=int)
            self.declare_partials(of=self.var_names[state_name]['k'],
                                  wrt=self.var_names[state_name]['f'],
                                  rows=ar, cols=ar)

            c = np.repeat(np.arange(num_steps, dtype=int), num_stages * size)
            self.declare_partials(of=self.var_names[state_name]['k'],
                                  wrt='h',
                                  rows=ar,
                                  cols=c)

    def compute(self, inputs, outputs):
        h = inputs['h']
        for state_name, options in iteritems(self.options['state_options']):
            f_transpose = inputs[self.var_names[state_name]['f']].T
            outputs[self.var_names[state_name]['k']] = (f_transpose * h).T

    def compute_partials(self, inputs, partials):
        h = inputs['h']

        for state_name, options in iteritems(self.options['state_options']):
            partials[self.var_names[state_name]['k'], self.var_names[state_name]['f']] = \
                np.repeat(h, self.options['num_steps'])

            partials[self.var_names[state_name]['k'], 'h'] = \
                inputs[self.var_names[state_name]['f']].ravel()