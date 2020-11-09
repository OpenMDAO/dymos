import numpy as np
from scipy.linalg import block_diag

import openmdao.api as om

from ....utils.rk_methods import rk_methods
from ....options import options as dymos_options


class RungeKuttaStatePredictComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase')

        self.options.declare('method', default='RK4', types=str,
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        self._var_names = {}
        num_seg = self.options['num_segments']

        rk_data = rk_methods[self.options['method']]
        self._A = block_diag(rk_data['A'])
        self._num_stages = rk_data['num_stages']

        for name, options in self.options['state_options'].items():
            shape = options['shape']
            units = options['units']

            self._var_names[name] = {}
            self._var_names[name]['initial'] = 'initial_states_per_seg:{0}'.format(name)
            self._var_names[name]['k'] = 'k:{0}'.format(name)
            self._var_names[name]['predicted'] = 'predicted_states:{0}'.format(name)

            self.add_input(self._var_names[name]['initial'], shape=(num_seg,) + shape, units=units,
                           desc='The initial value of the state at the start of the segment.')

            self.add_input(self._var_names[name]['k'],
                           shape=(num_seg, self._num_stages,) + shape,
                           units=units, desc='RK multiplier k for each stage in the segment.')

            self.add_output(self._var_names[name]['predicted'],
                            shape=(num_seg * self._num_stages,) + shape,
                            units=units,
                            desc='The predicted values of the state at the ODE evaluation points.')

            e = np.eye(np.prod(shape))
            p = np.kron(np.ones(self._num_stages), e).T
            p = block_diag(*num_seg*[p])
            r, c = np.nonzero(p)
            self.declare_partials(of=self._var_names[name]['predicted'],
                                  wrt=self._var_names[name]['initial'],
                                  rows=r, cols=c, val=1.0)

            size = np.prod(shape)
            p = block_diag(*num_seg*[np.kron(self._A, np.eye(size))])
            r, c = np.nonzero(p)
            self.declare_partials(of=self._var_names[name]['predicted'],
                                  wrt=self._var_names[name]['k'],
                                  rows=r, cols=c, val=p[r, c])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        #
        # Note:  To accommodate states of dimension 2 or more, k is flattened to
        # (num_segments * num_stage, size) prior to the matrix vector product,
        # and then reshaped back to the appropriate one.  This is complicated by the fact that
        # we need k in (num_stages, num_segments, size) for the matrix multiply to work out.
        # See below for the steps taken.
        #
        num_seg = self.options['num_segments']
        num_stages = self._num_stages
        for name, options in self.options['state_options'].items():
            shape = options['shape']
            size = np.prod(shape)
            x0 = inputs[self._var_names[name]['initial']]

            # # Reorder k to (num_stages, num_segments, size) to avoid a for loop
            # # k = np.moveaxis(inputs[self._var_names[name]['k']],
            # #                 [0, 1], [1, 0]).reshape((num_stages, size * num_seg))
            # # Einsum appears to be faster than moveaxis, so we use this implementation instead.
            k = np.einsum('ij...->ji...',
                          inputs[self._var_names[name]['k']]).reshape((num_stages, size * num_seg))

            # The matrix multiply
            A_dot_k = np.dot(self._A, k)

            # Now stack the segments vertically so we have num_seg * num_stages rows and size cols
            A_dot_k = np.vstack(np.split(A_dot_k, num_seg, axis=1))

            # Now give x0 the same treatment.
            # First broadcast to num_stages rows and num_segments * size cols
            _x0 = np.repeat(np.reshape(x0, (1, num_seg * size)), num_stages, axis=0)

            # Now chunk x0 into num_segments * num_stages rows and size cols
            _x0 = np.vstack(np.split(_x0, num_seg, axis=1))

            # Now compute the predicted state and change back to the correct shape at each row
            x_p = np.reshape(_x0 + A_dot_k, newshape=(num_seg * num_stages,) + shape)

            outputs[self._var_names[name]['predicted']] = x_p
