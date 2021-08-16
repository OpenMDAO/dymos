import numpy as np
import openmdao.api as om

from .control_interpolation_comp import ControlInterpolationComp
from .state_rate_collector_comp import StateRateCollectorComp
from .tau_comp import TauComp


class ODEEvaluationGroup(om.Group):
    """
    A special group whose purpose is to evaluate the ODE and return the computed
    state weights.
    """

    def __init__(self, ode_class, time_options, state_options, control_options,
                 polynomial_control_options, parameter_options, ode_init_kwargs=None,
                 grid_data=None, **kwargs):
        super().__init__(**kwargs)

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
        self.grid_data = grid_data
        self.ode_init_kwargs = {} if ode_init_kwargs is None else ode_init_kwargs

    def setup(self):
        gd = self.grid_data

        ### All times, states, controls, parameters, and polyomial controls need to exist
        # in the ODE evaluation group regardless of whether or not they have targets in the ODE.
        # This makes taking the derivatives more consistent without Exceptions.
        self._ivc = self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        if self.control_options or self.polynomial_control_options:
            c_options = self.control_options
            pc_options = self.polynomial_control_options

            # Add a component to compute the current non-dimensional phase time.
            self.add_subsystem('tau_comp', TauComp(grid_data=self.grid_data,
                                                   time_units=self.time_options['units']),
                               promotes_inputs=['time', 't_initial', 't_duration'],
                               promotes_outputs=['stau', 'ptau', 'time_phase', 'segment_index'])
            # Add control interpolant
            self._control_comp = self.add_subsystem('control_interp',
                                                    ControlInterpolationComp(grid_data=gd,
                                                                             control_options=c_options,
                                                                             polynomial_control_options=pc_options,
                                                                             time_units=self.time_options['units']))
        # # Required
        # self.options.declare('num_nodes', types=int,
        #                      desc='The total number of points at which times are required in the'
        #                           'phase.')
        #
        # self.options.declare('node_ptau', types=(np.ndarray,),
        #                      desc='The locations of all nodes in non-dimensional phase tau space.')
        #
        # self.options.declare('node_dptau_dstau', types=(np.ndarray,),
        #                      desc='For each node, the ratio of the total phase length to the length'
        #                           ' of the nodes containing segment.')
        #
        # # Optional
        # self.options.declare('units', default=None, allow_none=True, types=str,
        #                      desc='Units of time (or the integration variable)')
        #
        # self.options.declare('initial_val', default=0.0, types=(int, float),
        #                      desc='default value of initial time')
        #
        # self.options.declare('duration_val', default=1.0, types=(int, float),
        #                      desc='default value of duration')


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
        self._configure_controls()

    def _configure_time(self):
        targets = self.time_options['targets']
        time_phase_targets = self.time_options['time_phase_targets']
        t_initial_targets = self.time_options['t_initial_targets']
        t_duration_targets = self.time_options['t_duration_targets']
        units = self.time_options['units']

        for tgts, var in [(targets, 'time'), (time_phase_targets, 'time_phase'),
                          (t_initial_targets, 't_initial'), (t_duration_targets, 't_duration')]:
            if var != 'time_phase':
                self._ivc.add_output(var, shape=(1,), units=units)
            for t in tgts:
                self.promotes('ode', inputs=[(t, var)])
            if tgts:
                self.set_input_defaults(name=var,
                                        val=np.ones((1,)),
                                        units=units)

    def _configure_states(self):
        for name, options in self.state_options.items():
            shape = options['shape']
            units = options['units']
            targets = options['targets'] if options['targets'] is not None else []
            rate_path, rate_io = self._get_rate_source_path(name)
            var_name = f'states:{name}'

            self._ivc.add_output(var_name, shape=shape, units=units)
            self.add_design_var(var_name)

            # Promote targets from the ODE
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, var_name)])
            if targets:
                self.set_input_defaults(name=var_name,
                                        val=np.ones(shape),
                                        units=options['units'])

            # If the state rate source is an output, connect it, otherwise
            # promote it to the appropriate name
            if rate_io == 'output':
                self.connect(rate_path, f'state_rate_collector.state_rates_in:{name}_rate')
            else:
                self.promotes('state_rate_collector',
                              inputs=[(f'state_rates_in:{name}_rate', rate_path)])

            self.add_constraint(f'state_rate_collector.state_rates:{name}_rate')

    def _configure_params(self):
        for name, options in self.parameter_options.items():
            shape = options['shape']
            units = options['units']
            targets = options['targets']
            var_name = f'parameters:{name}'

            self._ivc.add_output(var_name, shape=shape, units=units)
            self.add_design_var(var_name)

            # Promote targets from the ODE
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, var_name)])
            if targets:
                self.set_input_defaults(name=var_name,
                                        val=np.ones(shape),
                                        units=options['units'])

    def _configure_controls(self):

        if self.control_options:
            if self.grid_data is None:
                raise ValueError('ODEEvaluationGroup was provided with control options but the '
                                 'a GridData object was not provided.')
            control_input_node_ptau = self.grid_data.node_ptau[
                self.grid_data.subset_node_indices['control_input']]
            num_control_input_nodes = len(control_input_node_ptau)
            # self.connect('tau_comp.ptau', 'control_interp.ptau')
            # self.connect('tau_comp.stau', 'control_interp.stau')
            # self.connect('tau_comp.segment_index', 'control_interp.segment_index')

            for name, options in self.control_options.items():
                shape = options['shape']
                units = options['units']
                targets = options['targets']
                uhat_name = f'controls:{name}'
                u_name = f'control_values:{name}'

                self._ivc.add_output(uhat_name, shape=(num_control_input_nodes,) + shape, units=units)
                self.add_design_var(uhat_name)

                self.promotes('control_interp', inputs=[uhat_name], outputs=[u_name])

                # Promote targets from the ODE
                for tgt in targets:
                    self.promotes('ode', inputs=[(tgt, u_name)])
                if targets:
                    self.set_input_defaults(name=u_name,
                                            val=np.ones(shape),
                                            units=options['units'])

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
