from __future__ import print_function, division, absolute_import

from six import string_types, iteritems

import numpy as np

from openmdao.api import ExplicitComponent

from ....utils.rk_methods import rk_methods


class RungeKuttaStatePredictComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('method', default='rk4', values=('rk4',),
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

    def setup(self):

        self._var_names = {}

        rk_data = rk_methods[self.options['method']]
        self._A = rk_data['A']
        self._num_stages = rk_data['num_stages']

        for name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']

            self._var_names[name] = {}
            self._var_names[name]['initial'] = 'initial_states:{0}'.format(name)
            self._var_names[name]['k'] = 'k:{0}'.format(name)
            self._var_names[name]['predicted'] = 'predicted_states:{0}'.format(name)

            self.add_input(self._var_names[name]['initial'], shape=shape, units=units,
                           desc='The initial value of the state at the start of the segment.')

            self.add_input(self._var_names[name]['k'], shape=(self._num_stages,) + shape,
                           units=units, desc='RK multiplier k for each stage in the segment.')

            self.add_output(self._var_names[name]['predicted'], shape=(self._num_stages,) + shape,
                            units=units,
                            desc='The predicted values of the state at the ODE evaluation points.')

            e = np.eye(np.prod(shape))
            p = np.kron(np.ones(self._num_stages), e).T
            r, c = np.nonzero(p)
            self.declare_partials(of=self._var_names[name]['predicted'],
                                  wrt=self._var_names[name]['initial'],
                                  rows=r, cols=c, val=1.0)

            self.declare_partials(of=self._var_names[name]['predicted'],
                                  wrt=self._var_names[name]['k'],
                                  val=np.kron(self._A, np.eye(np.prod(shape))))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        #
        # Note:  To accommodate states of dimension 2 or more, k is flattened to (num_stage, size)
        #        prior to the matrix vector product, and then reshaped back to the appropriate one.
        #
        num_stages = self._num_stages
        for name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            size = np.prod(options['shape'])
            x0 = inputs[self._var_names[name]['initial']]
            k = inputs[self._var_names[name]['k']].reshape((num_stages, size))
            out_shape = (num_stages,) + shape
            outputs[self._var_names[name]['predicted']] = x0 + np.dot(self._A, k).reshape(out_shape)
