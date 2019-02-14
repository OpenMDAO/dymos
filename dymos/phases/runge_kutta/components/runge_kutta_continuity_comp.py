"""Define the BalanceComp class."""

from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent


class RungeKuttaContinuityComp(ImplicitComponent):
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
        self._fixed_idx = {}

        for state_name, options in iteritems(state_options):

            if options['fix_initial'] == options['fix_final']:
                raise ValueError('RungeKuttaPhase requires that ONE of state options "fix_initial" '
                                 'or "fix_final" is True, but NOT BOTH. \nState: {0}, fix_initial: '
                                 '{1}, fix_final: {2}'.format(state_name, options['fix_initial'],
                                                              options['fix_final']))

            self._fixed_idx[state_name] = 0 if options['fix_initial'] else -1

            self._var_names[state_name] = {
                'seg_ends': 'states:{0}'.format(state_name),
                'initial': 'initial_states:{0}'.format(state_name),
                'final': 'final_states:{0}'.format(state_name),
            }

            shape = options['shape']
            units = options['units']
            var_names = self._var_names[state_name]

            # The implicit variable is the state values at all segment endpoints.
            self.add_output(name=var_names['seg_ends'],
                            shape=(num_seg + 1,) + shape,
                            units=units)

            # # The value of the state at the start of each segment.
            # self.add_input(name=var_names['initial'],
            #                shape=(num_seg,) + shape,
            #                desc='Value of the state at the start of each segment (step)',
            #                units=units)

            # The value of the state at the end of each segment.
            self.add_input(name=var_names['final'],
                           shape=(num_seg,) + shape,
                           desc='Value of the state at the end of each segment (step)',
                           units=units)

            c = np.arange(num_seg + 1, dtype=int)
            r = c
            self.declare_partials(of=var_names['seg_ends'], wrt=var_names['seg_ends'],
                                  rows=r, cols=c, val=-1.0)

            c = np.arange(num_seg, dtype=int)
            r = c + 1
            self.declare_partials(of=var_names['seg_ends'], wrt=var_names['final'],
                                  rows=r, cols=c, val=1.0)

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

            xf = inputs[names['final']]

            residuals[names['seg_ends']][0, ...] = 0
            residuals[names['seg_ends']][1:, ...] = xf - outputs[names['seg_ends']][1:, ...]
