from __future__ import print_function, division, absolute_import

from six import string_types, iteritems

import numpy as np

from openmdao.api import ExplicitComponent

from ....utils.rk_methods import rk_methods


class RungeKuttaStateAdvanceComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('method', default='rk4', values=('rk4',),
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

    def setup(self):

        self._var_names = {}
        self._bflat = {}
        self._k_shapes = {}

        rk_data = rk_methods[self.options['method']]
        num_stages = rk_data['num_stages']

        for name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            self._var_names[name] = {}
            self._var_names[name]['initial'] = 'initial_states:{0}'.format(name)
            self._var_names[name]['k'] = 'k:{0}'.format(name)
            self._var_names[name]['final'] = 'final_states:{0}'.format(name)

            self._bflat[name] = np.repeat(rk_data['b'], np.prod(shape))
            self._k_shapes[name] = (num_stages,) + shape

            self.add_input(self._var_names[name]['initial'], shape=shape, units=units,
                           desc='The initial value of the state at the start of the segment.')

            self.add_input(self._var_names[name]['k'], shape=(num_stages,) + shape, units=units,
                           desc='RK multiplier k for each stage in the segment.')

            self.add_output(self._var_names[name]['final'], shape=shape,
                            units=units,
                            desc='The final value of the state at the end of the segment.')

            e = np.eye(size)
            r, c = np.nonzero(e)
            self.declare_partials(of=self._var_names[name]['final'],
                                  wrt=self._var_names[name]['initial'],
                                  rows=r, cols=c, val=1.0)

            p = np.kron(rk_data['b'], np.eye(size))
            r, c = np.nonzero(p)
            self.declare_partials(of=self._var_names[name]['final'],
                                  wrt=self._var_names[name]['k'],
                                  rows=r, cols=c,
                                  val=np.tile(rk_data['b'], size))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for name, options in iteritems(self.options['state_options']):
            x0 = inputs[self._var_names[name]['initial']]
            k = inputs[self._var_names[name]['k']]
            b = self._bflat[name]
            kshape = self._k_shapes[name]
            out_name = self._var_names[name]['final']
            outputs[out_name] = x0 + np.sum((k.ravel() * b).reshape(kshape), axis=0)
