"""Define the BalanceComp class."""

from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent


class RungeKuttaStateContinuityComp(ImplicitComponent):
    """
    A simple equation balance for solving implicit equations.
    Attributes
    ----------
    _state_vars : dict
        Cache the data provided during `add_balance`
        so everything can be saved until setup is called.
    """

    def initialize(self):

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase.')

        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase.')

    def setup(self):
        """
        Define the independent variables, output variables, and partials.
        """
        num_seg = self.options['num_segments']
        state_options = self.options['state_options']

        self._var_names = {}

        for state_name, options in iteritems(state_options):

            self._var_names[state_name] = {
                'states': 'states:{0}'.format(state_name),
                'integral': 'state_integrals:{0}'.format(state_name)
            }

            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            var_names = self._var_names[state_name]

            # The implicit variable is the state values at all segment endpoints.
            self.add_output(name=var_names['states'],
                            shape=(num_seg + 1,) + shape,
                            units=units)

            # The value of the state at the end of each segment.
            self.add_input(name=var_names['integral'],
                           shape=(num_seg,) + shape,
                           desc='Change in the state value over each segment.',
                           units=units)

            #
            # Define the partials of the states wrt themselves.
            #
            n_rows = size * (1 + num_seg)
            n_cols = n_rows
            temp_jac = np.zeros((n_rows, n_cols), dtype=int)
            eye_size = np.eye(size, dtype=int)
            pattern = np.zeros((num_seg, num_seg + 1))
            diag_rows, diag_cols = np.diag_indices(num_seg)
            pattern[diag_rows, diag_cols] = -1
            pattern[diag_rows, diag_cols + 1] = 1
            temp_jac[size:, :] = np.kron(pattern, eye_size)
            temp_jac[:size, :size] = -np.eye(size)
            r, c = np.nonzero(temp_jac)
            v = temp_jac[r, c]
            self.declare_partials(of=var_names['states'], wrt=var_names['states'],
                                  rows=r, cols=c, val=v)

            #
            # Define the partials of the states wrt the state integrals.
            #
            n_rows = size * (1 + num_seg)
            n_cols = size * num_seg
            temp_jac = np.zeros((n_rows, n_cols), dtype=int)
            temp_jac[size:, :] = np.eye(n_cols, dtype=int)
            r, c = np.nonzero(temp_jac)
            self.declare_partials(of=var_names['states'], wrt=var_names['integral'],
                                  rows=r, cols=c, val=-1.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Calculate the residual for each state value.
        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        state_options = self.options['state_options']

        for state_name, options in iteritems(state_options):
            names = self._var_names[state_name]
            # direction = options['time_direction']

            x_i = outputs[names['states']][:-1, ...]
            x_f = outputs[names['states']][1:, ...]
            dx = inputs[names['integral']]

            # if direction == 'forward':
            residuals[names['states']][0, ...] = 0
            residuals[names['states']][1:, ...] = x_f - x_i - dx
            # else:
            #     residuals[names['states']][:-1, ...] = x_f - x_i - dx
            #     residuals[names['states']][-1, ...] = 0
