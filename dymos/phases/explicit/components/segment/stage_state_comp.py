from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import ExplicitComponent

from dymos.utils.rk_methods import rk_methods


class StageStateComp(ExplicitComponent):
    """
    Computes the values of the states to pass to the ODE for a given stage
    """

    def initialize(self):
        self.options.declare('num_steps', types=int)
        self.options.declare('state_options', types=dict)
        self.options.declare('method', types=str, default='rk4')
        self.var_names = {}

    def setup(self):
        method = self.options['method']
        num_steps = self.options['num_steps']
        num_stages = rk_methods[method]['num_stages']

        for state_name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']

            self.var_names[state_name] = {}

            self.var_names[state_name]['step_vals'] = 'step_states:{0}'.format(state_name)

            self.var_names[state_name]['k'] = 'k:{0}'.format(state_name)

            self.var_names[state_name]['stage_vals'] = 'stage_states:{0}'.format(state_name)

            self.add_input(name=self.var_names[state_name]['step_vals'],
                           shape=tuple([num_steps + 1] + list(shape)),
                           desc='State value at the step boundaries',
                           units=units)

            self.add_input(name=self.var_names[state_name]['k'],
                           shape=tuple([num_steps, num_stages] + list(shape)),
                           desc='Value of k computed for the previous stage',
                           units=units)

            self.add_output(name=self.var_names[state_name]['stage_vals'],
                            shape=tuple([num_steps, num_stages] + list(shape)),
                            desc='State value at the stages of each step',
                            units=units)

            self.declare_partials(of=self.var_names[state_name]['stage_vals'],
                                  wrt=self.var_names[state_name]['step_vals'],
                                  method='fd')

            self.declare_partials(of=self.var_names[state_name]['stage_vals'],
                                  wrt=self.var_names[state_name]['k'],
                                  method='fd')

    # def Y_calc_vec(y, K_flat):
    #     """ given the predicted increments (K),
    #         compute the predicted stage values (Y)
    #
    #         params
    #         -------
    #         y: (N+1,) float array
    #         K: (N, 4) float array
    #
    #         returns
    #         -------
    #         (N,4) float array
    #     """
    #
    #     N_STEPS = y.shape[0] - 1
    #     K = K_flat.reshape((N_STEPS, 4))
    #
    #     Y = np.zeros((N_STEPS, 4))
    #     for i in range(N_STEPS):
    #         K_ = np.zeros(4)
    #         K_[1:] = K[i, :3]
    #         Y[i, :] = y[i] + K_ * np.array([0.0, 0.5, 0.5, 1.0])
    #
    #     return Y

    def compute(self, inputs, outputs):
        method = self.options['method']
        num_steps = self.options['num_steps']
        num_stages = rk_methods[method]['num_stages']
        A = np.array([0.0, 0.5, 0.5, 1.0])

        # N_STEPS = y.shape[0] - 1
        # K = K_flat.reshape((N_STEPS, 4))
        # A_vec = np.array([0.0, 0.5, 0.5, 1.0])
        #
        # Y = np.zeros((N_STEPS, 4))
        # for i in range(N_STEPS):
        #     for j in range(1, 4):
        #         Y[i, j] = K[i, j - 1] * A_vec[j]
        #     Y[i] += y[i]

        for state_name, options in iteritems(self.options['state_options']):
            y_step = inputs[self.var_names[state_name]['step_vals']]
            y_stages = outputs[self.var_names[state_name]['stage_vals']]
            k = inputs[self.var_names[state_name]['k']].reshape(4, 4)

            y_stages[...] = 0.0
            for istep in range(num_steps):
                for jstage in range(1, num_stages):
                    y_stages[istep, jstage] = k[istep, jstage - 1] * A[jstage]
                y_stages[istep] += y_step[istep]


            # \y_stages[...] = y_step[:-1, ...] + np.dot(A, k)
