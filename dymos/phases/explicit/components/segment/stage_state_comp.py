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

        self.a = np.array([0.0, 0.5, 0.5, 1.0])

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
                           shape=tuple([num_steps] + [num_stages] + list(shape)),
                           desc='Value of k computed for the previous stage',
                           units=units)

            self.add_output(name=self.var_names[state_name]['stage_vals'],
                            shape=tuple([num_steps] + [num_stages] + list(shape)),
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

        for istep in range(num_steps):
            for state_name, options in iteritems(self.options['state_options']):
                y_step = inputs[self.var_names[state_name]['step_vals']]
                k = inputs[self.var_names[state_name]['k']]
                outputs[self.var_names[state_name]['stage_vals']][istep, :] = y_step[istep] + np.dot(self.a, k)

    # def compute_partials(self, inputs, partials):
    #     A = rk_methods[self.options['method']]['A']
    #     s = self.options['stage']
    #
    #     for state_name, options in iteritems(self.options['state_options']):
    #         of = self.var_names[state_name]['yf']
    #         k_name = self.var_names[state_name]['k']
    #         partials[of, k_name] = A[s - 1, s - 2]