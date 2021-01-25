"""Define the BalanceComp class."""

import numpy as np

from openmdao.core.implicitcomponent import ImplicitComponent
from ....options import options as dymos_options


class RungeKuttaStateContinuityComp(ImplicitComponent):
    """
    Class definition for the RungeKuttaStateContinuityComp.

    Implicitly solve the RungeKutta state continuity by forcing final state values to
    equal initial state values plus the state integral over each segment.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    Attributes
    ----------
    _var_names : dict
        Cache the data provided during `add_balance` so everything can be saved until setup is called.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase.')

        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase.')

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        num_seg = self.options['num_segments']
        state_options = self.options['state_options']

        self._var_names = {}

        for state_name, options in state_options.items():
            self._var_names[state_name] = {
                'initial': 'initial_states:{0}'.format(state_name),
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
                            units=units,
                            lower=options['lower'],
                            upper=options['upper'])

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

            # The initial value of the state at the start of the phase, if connected_initial == True
            if options['connected_initial']:
                self.add_input(name=var_names['initial'], shape=(1,) + shape, units=units)

                ar = np.arange(size, dtype=int)
                self.declare_partials(of=var_names['states'], wrt=var_names['initial'],
                                      rows=ar, cols=ar, val=1.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute residuals given inputs and outputs.

        The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        """
        state_options = self.options['state_options']

        for state_name, options in state_options.items():
            names = self._var_names[state_name]

            x_i = outputs[names['states']][:-1, ...]
            x_f = outputs[names['states']][1:, ...]
            dx = inputs[names['integral']]

            if options['connected_initial']:
                residuals[names['states']][0, ...] = \
                    inputs[names['initial']][0, ...] - outputs[names['states']][0, ...]

            else:
                residuals[names['states']][0, ...] = 0
            residuals[names['states']][1:, ...] = x_f - x_i - dx
