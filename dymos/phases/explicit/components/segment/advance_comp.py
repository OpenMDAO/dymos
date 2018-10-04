from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import ExplicitComponent

from dymos.utils.rk_methods import rk_methods


class AdvanceComp(ExplicitComponent):
    """
    Computes the new values of the states at the end of each RK Step
    """

    def initialize(self):
        self.options.declare('state_options', types=dict)
        self.options.declare('method', values=('rk4',), default='rk4')
        self.options.declare('num_steps', types=int)

        self.var_names = {}

    def setup(self):
        num_stages = rk_methods[self.options['method']]['num_stages']
        b = rk_methods[self.options['method']]['b']
        num_steps = self.options['num_steps']

        for state_name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            self.var_names[state_name] = {}
            self.var_names[state_name]['k'] = 'k:{0}'.format(state_name)
            self.var_names[state_name]['y0'] = 'initial_states:{0}'.format(state_name)
            self.var_names[state_name]['y'] = 'step_states:{0}'.format(state_name)

            self.add_input(name=self.var_names[state_name]['k'],
                           shape=tuple([num_steps, num_stages] + list(shape)),
                           desc='Value of k computed for each step/stage',
                           units=units)

            self.add_input(name=self.var_names[state_name]['y0'],
                           shape=shape,
                           desc='State value at the start of the segment'.format(state_name),
                           units=units)

            self.add_output(name=self.var_names[state_name]['y'],
                            shape=tuple([num_steps + 1] + list(shape)),
                            desc='State value at the end of each step'.format(state_name),
                            units=units)

            r = np.arange(1, num_steps + 1, dtype=int)
            r = np.repeat(r, np.arange(1, num_steps + 1, dtype=int))
            r = np.repeat(r, num_stages)

            c = []
            for i in range(1, num_steps + 1):
                c += np.arange(i * num_stages, dtype=int).tolist()

            repeats = len(c) // num_stages

            self.declare_partials(of=self.var_names[state_name]['y'],
                                  wrt=self.var_names[state_name]['k'],
                                  rows=r,
                                  cols=c,
                                  val=np.tile(b, repeats))

            self.declare_partials(of=self.var_names[state_name]['y'],
                                  wrt=self.var_names[state_name]['y0'],
                                  val=1.0)

    def compute(self, inputs, outputs):
        num_steps = self.options['num_steps']
        b = rk_methods[self.options['method']]['b']

        for state_name, options in iteritems(self.options['state_options']):
            y0 = inputs[self.var_names[state_name]['y0']]
            y = outputs[self.var_names[state_name]['y']].reshape(num_steps + 1)
            k = inputs[self.var_names[state_name]['k']].reshape((num_steps, 4))

            y[...] = 0.0

            np.einsum('j,ij->i', b, k, out=y[1:])
            np.cumsum(y[1:], out=y[1:])

            y += y0
