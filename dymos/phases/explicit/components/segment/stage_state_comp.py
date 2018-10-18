from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np
from scipy.linalg import block_diag

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
        self._state_sizes = {}

    def setup(self):
        method = self.options['method']
        num_steps = self.options['num_steps']
        num_stages = rk_methods[method]['num_stages']
        A = rk_methods[method]['A']

        # TODO: These derivatives are incompatible with nonscalar states
        d_dk = block_diag(*num_steps*[A])
        r_k, c_k = np.nonzero(d_dk)
        d_dk_vals = d_dk[r_k, c_k]

        c_s = np.repeat(np.arange(num_steps, dtype=int), num_stages)
        r_s = np.arange(num_steps * num_stages)

        for state_name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']
            self._state_sizes[state_name] = np.prod(shape)

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
                                  rows=r_s, cols=c_s, val=1.0)

            self.declare_partials(of=self.var_names[state_name]['stage_vals'],
                                  wrt=self.var_names[state_name]['k'],
                                  rows=r_k, cols=c_k, val=d_dk_vals)

    def compute(self, inputs, outputs):
        method = self.options['method']
        num_steps = self.options['num_steps']
        A = rk_methods[method]['A']

        for state_name, options in iteritems(self.options['state_options']):
            y_step = inputs[self.var_names[state_name]['step_vals']]
            y_stages = outputs[self.var_names[state_name]['stage_vals']]
            k = inputs[self.var_names[state_name]['k']]

            np.einsum('nij,njk->nik', A[np.newaxis, :, :], k, out=y_stages)
            for i in range(num_steps):
                y_stages[i, ...] += y_step[i]
