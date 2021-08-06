import numpy as np
import openmdao.api as om

from .state_rate_collector_comp import StateRateCollectorComp


class ODEEvaluationGroup(om.Group):
    """
    A special group whose purpose is to evaluate the ODE and return the computed
    state weights.
    """

    def __init__(self, ode_class, time_options, state_options, control_options,
                 polynomial_control_options, parameter_options, ode_init_kwargs=None):
        super().__init__()

        # Get the state vector.  This isn't necessarily ordered
        # so just pick the default ordering and go with it.
        self.state_options = state_options
        self.time_options = time_options
        self.control_options = control_options
        self.polynomial_control_options = polynomial_control_options
        self.parameter_options = parameter_options
        self.control_interpolants = {}
        self.polynomial_control_interpolants = {}
        self.ode_class = ode_class
        self.ode_init_kwargs = {} if ode_init_kwargs is None else ode_init_kwargs

    def setup(self):
        if self.control_options:
            # Add control interpolant
            raise NotImplementedError('dynamic controls not yet implemented')
        if self.polynomial_control_options:
            # Add polynomial control interpolant
            raise NotImplementedError('polynomial controls not yet implemented')

        self.add_subsystem('ode', self.ode_class(num_nodes=1, **self.ode_init_kwargs))

        self.add_subsystem('state_rate_collector',
                           StateRateCollectorComp(state_options=self.state_options,
                                                  time_units=self.time_options['units']))

    def configure(self):
        self._configure_time()
        self._configure_states()
        self._configure_params()

    def _configure_time(self):
        targets = self.time_options['targets']
        time_phase_targets = self.time_options['time_phase_targets']
        t_initial_targets = self.time_options['t_initial_targets']
        t_duration_targets = self.time_options['t_duration_targets']
        units = self.time_options['units']

        for tgts, var in [(targets, 'time'), (time_phase_targets, 'time_phase'),
                          (t_initial_targets, 't_initial'), (t_duration_targets, 't_duration')]:
            for t in tgts:
                self.promotes('ode', inputs=[(t, var)])
            if tgts:
                self.set_input_defaults(name=var,
                                        val=np.ones((1,)),
                                        units=units)

    def _configure_states(self):
        for name, options in self.state_options.items():
            shape = options['shape']
            targets = options['targets']
            rate_path, rate_io = self._get_rate_source_path(name)

            # Promote targets from the ODE
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, f'states:{name}')])
            if targets:
                self.set_input_defaults(name=f'states:{name}',
                                        val=np.ones(shape),
                                        units=options['units'])

            # If the state rate source is an output, connect it, otherwise
            # promote it to the appropriate name
            if rate_io == 'output':
                self.connect(rate_path, f'state_rate_collector.state_rates_in:{name}_rate')
            else:
                self.promotes('state_rate_collector',
                              inputs=[(f'state_rates_in:{name}_rate', rate_path)])

            self.add_design_var(f'states:{name}')
            self.add_constraint(f'state_rate_collector.state_rates:{name}_rate')

    def _configure_params(self):
        for name, options in self.parameter_options.items():
            shape = options['shape']
            targets = options['targets']
            # rate_path, rate_io = self._get_rate_source_path(name)

            # Promote targets from the ODE
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, f'parameters:{name}')])
            if targets:
                self.set_input_defaults(name=f'parameters:{name}',
                                        val=np.ones(shape),
                                        units=options['units'])

            self.add_design_var(f'parameters:{name}')

    def _get_rate_source_path(self, state_var):
        """
        Get path of the rate source variable so that we can connect it to the
        outputs when we're done.

        Parameters
        ----------
        state_var : str
            The name of the state variable whose path is desired.

        Returns
        -------
        path : str
            The path to the rate source of the state variable.
        io : str
            A string indicating whether the variable in the path is an 'input'
            or an 'output'.
        """
        var = self.state_options[state_var]['rate_source']

        if var == 'time':
            rate_path = 'time'
            io = 'input'
        elif var == 'time_phase':
            rate_path = 'time_phase'
            io = 'input'
        elif self.state_options is not None and var in self.state_options:
            rate_path = f'states:{var}'
            io = 'input'
        elif self.control_options is not None and var in self.control_options:
            rate_path = f'controls:{var}'
            io = 'output'
        elif self.polynomial_control_options is not None and var in self.polynomial_control_options:
            rate_path = f'polynomial_controls:{var}'
            io = 'output'
        elif self.parameter_options is not None and var in self.parameter_options:
            rate_path = f'parameters:{var}'
            io = 'input'
        elif var.endswith('_rate') and self.control_options is not None and \
                var[:-5] in self.control_options:
            rate_path = f'control_rates:{var}'
            io = 'output'
        elif var.endswith('_rate2') and self.control_options is not None and \
                var[:-6] in self.control_options:
            rate_path = f'control_rates:{var}'
            io = 'output'
        elif var.endswith('_rate') and self.polynomial_control_options is not None and \
                var[:-5] in self.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
            io = 'output'
        elif var.endswith('_rate2') and self.polynomial_control_options is not None and \
                var[:-6] in self.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
            io = 'output'
        else:
            rate_path = f'ode.{var}'
            io = 'output'
        return rate_path, io
