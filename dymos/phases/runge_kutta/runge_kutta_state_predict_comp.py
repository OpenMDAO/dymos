from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np
from scipy.linalg import block_diag

from openmdao.api import ExplicitComponent

from ...utils.rk_methods import rk_methods


class RungeKuttaStatePredictComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase')

        self.options.declare('method', default='rk4', values=('rk4',),
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

    def setup(self):

        self._var_names = {}
        num_seg = self.options['num_segments']

        rk_data = rk_methods[self.options['method']]
        # print(num_seg * [rk_data['A']])
        # self._A = block_diag(num_seg * [rk_data['A']])
        # self._A = np.block([[rk_data['A'], rk_data['A'], rk_data['A'], rk_data['A']],
        #                     [rk_data['A'], rk_data['A'], rk_data['A'], rk_data['A']],
        #                     [rk_data['A'], rk_data['A'], rk_data['A'], rk_data['A']],
        #                     [rk_data['A'], rk_data['A'], rk_data['A'], rk_data['A']]])
        # self._A = block_diag(*4*[rk_data['A']])
        self._A = block_diag(rk_data['A'])
        self._num_stages = rk_data['num_stages']

        for name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']

            self._var_names[name] = {}
            self._var_names[name]['initial'] = 'initial_states:{0}'.format(name)
            self._var_names[name]['k'] = 'k:{0}'.format(name)
            self._var_names[name]['predicted'] = 'predicted_states:{0}'.format(name)

            self.add_input(self._var_names[name]['initial'], shape=(num_seg,) + shape, units=units,
                           desc='The initial value of the state at the start of the segment.')

            self.add_input(self._var_names[name]['k'],
                           shape=(num_seg, self._num_stages,) + shape,
                           units=units, desc='RK multiplier k for each stage in the segment.')

            self.add_output(self._var_names[name]['predicted'],
                            shape=(num_seg, self._num_stages,) + shape,
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
        # Note:  To accommodate states of dimension 2 or more, k is flattened to (num_segments * num_stage, size)
        #        prior to the matrix vector product, and then reshaped back to the appropriate one.
        #
        num_seg = self.options['num_segments']
        num_stages = self._num_stages
        for name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            size = np.prod(options['shape'])
            x0 = inputs[self._var_names[name]['initial']]

            for i in range(num_seg):
                k = inputs[self._var_names[name]['k']][i, ...].reshape((num_stages, size))
                out_shape = (num_stages,) + shape
                outputs[self._var_names[name]['predicted']][i, ...] = x0 + np.dot(self._A, k).reshape(out_shape)
