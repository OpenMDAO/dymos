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
            direction = options['time_direction']

            self._var_names[state_name] = {
                'states': 'states:{0}'.format(state_name),
                'integral': 'state_integrals:{0}'.format(state_name)
            }

            shape = options['shape']
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

            if direction == 'forward':
                r = np.repeat(np.arange(1, num_seg + 1, dtype=int), repeats=2)
                r = np.concatenate(([0], r))
                c = np.repeat(np.arange(num_seg, dtype=int), repeats=2)
                c = np.concatenate((c, [num_seg]))
                v = np.tile(np.array([1, -1]), num_seg)
                v = np.concatenate((v, [1]))
            else:
                r = np.repeat(np.arange(num_seg, dtype=int), repeats=2)
                r = np.concatenate((r, [num_seg]))
                c = np.repeat(np.arange(1, num_seg + 1, dtype=int), repeats=2)
                c = np.concatenate(([0], c))
                v = np.tile(np.array([-1, 1]), num_seg)
                v = np.concatenate((v, [-1]))

            self.declare_partials(of=var_names['states'], wrt=var_names['states'],
                                  rows=r, cols=c, val=v)

            if direction == 'forward':
                c = np.arange(num_seg, dtype=int)
                r = c + 1
                self.declare_partials(of=var_names['states'], wrt=var_names['integral'],
                                      rows=r, cols=c, val=-1.0)
            else:
                c = np.arange(num_seg, dtype=int)
                r = c
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
            direction = options['time_direction']

            x_i = outputs[names['states']][:-1, ...]
            x_f = outputs[names['states']][1:, ...]
            dx = inputs[names['integral']]

            # Currently the direction of propagation is set at the phase level.  We can
            # compute this on a state-by-state basis if necessary.

            if direction == 'forward':
                residuals[names['states']][0, ...] = 0
                residuals[names['states']][1:, ...] = x_f - x_i - dx
            else:
                residuals[names['states']][:-1, ...] = x_f - x_i - dx
                residuals[names['states']][-1, ...] = 0
