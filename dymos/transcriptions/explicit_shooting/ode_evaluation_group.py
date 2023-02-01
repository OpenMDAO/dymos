import numpy as np
import openmdao.api as om

from .vandermonde_control_interp_comp import VandermondeControlInterpComp
from .state_rate_collector_comp import StateRateCollectorComp
from .tau_comp import TauComp

from ...utils.introspection import get_targets, configure_controls_introspection,\
    configure_time_introspection, configure_parameters_introspection, \
    configure_states_discovery, configure_states_introspection, get_target_metadata
from ...utils.misc import get_rate_units


class ODEEvaluationGroup(om.Group):
    """
    A group whose purpose is to evaluate the ODE and output the computed state rates.

    Parameters
    ----------
    ode_class : class
        The class of the OpenMDAO system to be used to evaluate the ODE in this Group.
    input_grid_data : GridData
        The GridData used to define the controls used in this ODE.
    time_options : OptionsDictionary
        OptionsDictionary of time options.
    state_options : dict of {str: OptionsDictionary}
        For each state variable, a dictionary of its options, keyed by name.
    parameter_options : dict of {str: OptionsDictionary}
        For each parameter, a dictionary of its options, keyed by name.
    control_options : dict of {str: OptionsDictionary}
        For each control variable, a dictionary of its options, keyed by name.
    polynomial_control_options : dict of {str: OptionsDictionary}
        For each polynomial variable, a dictionary of its options, keyed by name.
    ode_init_kwargs : dict
        A dictionary of keyword arguments to be passed to the instantiation of the ODE.
    vec_size : int
        The number of points at which the ODE is simultaneously evaluated.
    **kwargs : dict
        Additional keyword arguments passed to Group.
    """

    def __init__(self, ode_class, input_grid_data, time_options, state_options, parameter_options, control_options,
                 polynomial_control_options, ode_init_kwargs=None, vec_size=1, **kwargs):
        super().__init__(**kwargs)

        self._state_options = state_options
        self._parameter_options = parameter_options
        self._time_options = time_options
        self._control_options = control_options
        self._polynomial_control_options = polynomial_control_options
        self._control_interpolants = {}
        self._polynomial_control_interpolants = {}
        self._ode_class = ode_class
        self._input_grid_data = input_grid_data
        self._vec_size = vec_size
        self._ode_init_kwargs = {} if ode_init_kwargs is None else ode_init_kwargs

    def set_segment_index(self, seg_idx):
        """
        Set the segment_index option on those subsystems which require it.

        Parameters
        ----------
        seg_idx : int
            The index of the current segment.
        """
        self._get_subsystem('tau_comp').options['segment_index'] = seg_idx

        control_interp_comp = self._get_subsystem('control_interp')
        if control_interp_comp:
            control_interp_comp.options['segment_index'] = seg_idx

    def setup(self):
        """
        Define the structure of the ODEEvaluationGroup.
        """
        igd = self._input_grid_data
        t_name = self._time_options['name']
        t_units = self._time_options['units']

        # All states, controls, parameters, and polynomial controls need to exist
        # in the ODE evaluation group regardless of whether or not they have targets in the ODE.
        # This makes taking the derivatives more consistent without Exceptions.
        self._ivc = self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        # Add a component to compute the current non-dimensional phase time.
        self.add_subsystem('tau_comp',
                           TauComp(grid_data=self._input_grid_data, vec_size=self._vec_size, time_units=t_units),
                           promotes_inputs=[('t', t_name), 't_initial', 't_duration'],
                           promotes_outputs=['stau', 'ptau', 'dstau_dt', ('t_phase', f'{t_name}_phase')])

        if self._control_options or self._polynomial_control_options:
            c_options = self._control_options
            pc_options = self._polynomial_control_options

            # Add control interpolant
            self._control_comp = self.add_subsystem('control_interp',
                                                    VandermondeControlInterpComp(grid_data=igd,
                                                                                 vec_size=self._vec_size,
                                                                                 control_options=c_options,
                                                                                 polynomial_control_options=pc_options,
                                                                                 time_units=t_units),
                                                    promotes_inputs=['ptau', 'stau', 't_duration', 'dstau_dt'])

        self.add_subsystem('ode', self._ode_class(num_nodes=self._vec_size, **self._ode_init_kwargs))

        self.add_subsystem('state_rate_collector',
                           StateRateCollectorComp(state_options=self._state_options,
                                                  time_units=self._time_options['units'],
                                                  vec_size=self._vec_size))

    def configure(self):
        """
        Perform I/O creation for this group's underlying members.

        In dymos, this system sits within a subproblem and therefore isn't in the standard
        configuration chain.  We need to perform all of the introspection of the ODE here.
        """
        ode = self._get_subsystem('ode')

        configure_time_introspection(self._time_options, ode)
        self._configure_time()

        configure_parameters_introspection(self._parameter_options, ode)
        self._configure_params()

        configure_controls_introspection(self._control_options, ode,
                                         time_units=self._time_options['units'])
        self._configure_controls()

        configure_controls_introspection(self._polynomial_control_options, ode,
                                         time_units=self._time_options['units'])
        self._configure_polynomial_controls()

        if self._control_options or self._polynomial_control_options:
            self._get_subsystem('control_interp').configure_io()

        configure_states_discovery(self._state_options, ode)
        configure_states_introspection(self._state_options, self._time_options, self._control_options,
                                       self._parameter_options,
                                       self._polynomial_control_options, ode)
        self._configure_states()

        self.state_rate_collector.configure_io()

    def _configure_time(self):
        vec_size = self._vec_size
        t_name = self._time_options['name']
        targets = self._time_options['targets']
        t_phase_targets = self._time_options['time_phase_targets']
        t_initial_targets = self._time_options['t_initial_targets']
        t_duration_targets = self._time_options['t_duration_targets']
        units = self._time_options['units']

        self._ivc.add_output(t_name, shape=(vec_size,), units=units)
        self._ivc.add_output('t_initial', shape=(1,), units=units)
        self._ivc.add_output('t_duration', shape=(1,), units=units)

        self.add_design_var(t_name)
        self.add_design_var('t_initial')
        self.add_design_var('t_duration')

        for tgts, var in [(targets, t_name), (t_phase_targets, f'{t_name}_phase'),
                          (t_initial_targets, 't_initial'), (t_duration_targets, 't_duration')]:
            for t in tgts:
                self.promotes('ode', inputs=[(t, var)])
            if tgts:
                self.set_input_defaults(name=var,
                                        val=np.ones((1,)),
                                        units=units)

    def _configure_states(self):
        vec_size = self._vec_size

        for name, options in self._state_options.items():
            shape = options['shape']
            units = options['units']
            targets = options['targets'] if options['targets'] is not None else []
            rate_path = self._get_rate_source_path(name)
            var_name = f'states:{name}'

            self._ivc.add_output(var_name, shape=(vec_size,) + shape, units=units)
            self.add_design_var(var_name)

            # Promote targets from the ODE
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, var_name)])
            if targets:
                self.set_input_defaults(name=var_name,
                                        val=np.ones(shape),
                                        units=options['units'])

            self.connect(rate_path, f'state_rate_collector.state_rates_in:{name}_rate')

            self.add_constraint(f'state_rate_collector.state_rates:{name}_rate')

    def _configure_params(self):
        vec_size = self._vec_size

        for name, options in self._parameter_options.items():
            var_name = f'parameters:{name}'

            targets = get_targets(ode=self.ode, name=name, user_targets=options['targets'])

            shape, units, static = get_target_metadata(self.ode, name=name,
                                                       user_targets=targets,
                                                       user_shape=options['shape'],
                                                       user_units=options['units'],
                                                       user_static_target=options['static_target'])
            options['units'] = units
            options['shape'] = shape
            options['static_target'] = static

            self._ivc.add_output(var_name, shape=shape, units=units)
            self.add_design_var(var_name)

            if options['static_target']:
                src_idxs = None
            else:
                src_rows = np.zeros(vec_size, dtype=int)
                src_idxs = om.slicer[src_rows, ...]

            # Promote targets from the ODE
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, var_name)], src_indices=src_idxs,
                              src_shape=options['shape'])
            if targets:
                self.set_input_defaults(name=var_name,
                                        val=np.ones(shape),
                                        units=options['units'])

    def _configure_controls(self):
        configure_controls_introspection(self._control_options, self.ode)
        time_units = self._time_options['units']

        if self._control_options:
            igd = self._input_grid_data

            if igd is None:
                raise ValueError('ODEEvaluationGroup was provided with control options but '
                                 'a GridData object was not provided.')

            num_control_input_nodes = igd.subset_num_nodes['control_input']

            for name, options in self._control_options.items():
                shape = options['shape']
                units = options['units']
                rate_units = get_rate_units(units, time_units, deriv=1)
                rate2_units = get_rate_units(units, time_units, deriv=2)
                targets = options['targets']
                rate_targets = options['rate_targets']
                rate2_targets = options['rate2_targets']
                uhat_name = f'controls:{name}'
                u_name = f'control_values:{name}'
                u_rate_name = f'control_rates:{name}_rate'
                u_rate2_name = f'control_rates:{name}_rate2'

                self._ivc.add_output(uhat_name, shape=(num_control_input_nodes,) + shape, units=units)
                self.add_design_var(uhat_name)
                self.add_constraint(u_name)
                self.add_constraint(u_rate_name)
                self.add_constraint(u_rate2_name)

                self.promotes('control_interp', inputs=[uhat_name],
                              outputs=[u_name, u_rate_name, u_rate2_name])

                # Promote targets from the ODE
                for tgt in targets:
                    self.promotes('ode', inputs=[(tgt, u_name)])
                if targets:
                    self.set_input_defaults(name=u_name,
                                            val=np.ones(shape),
                                            units=options['units'])

                # Promote rate targets from the ODE
                for tgt in rate_targets:
                    self.promotes('ode', inputs=[(tgt, u_rate_name)])
                if rate_targets:
                    self.set_input_defaults(name=u_rate_name,
                                            val=np.ones(shape),
                                            units=rate_units)

                # Promote rate2 targets from the ODE
                for tgt in rate2_targets:
                    self.promotes('ode', inputs=[(tgt, u_rate2_name)])
                if rate2_targets:
                    self.set_input_defaults(name=u_rate2_name,
                                            val=np.ones(shape),
                                            units=rate2_units)

    def _configure_polynomial_controls(self):
        configure_controls_introspection(self._polynomial_control_options, self.ode)

        if self._polynomial_control_options:
            time_units = self._time_options['units']
            gd = self._input_grid_data

            if gd is None:
                raise ValueError('ODEEvaluationGroup was provided with control options but '
                                 'a GridData object was not provided.')

            for name, options in self._polynomial_control_options.items():
                shape = options['shape']
                units = options['units']
                rate_units = get_rate_units(units, time_units, deriv=1)
                rate2_units = get_rate_units(units, time_units, deriv=2)
                targets = options['targets']
                rate_targets = options['rate_targets']
                rate2_targets = options['rate2_targets']
                num_control_input_nodes = options['order'] + 1
                uhat_name = f'polynomial_controls:{name}'
                u_name = f'polynomial_control_values:{name}'
                u_rate_name = f'polynomial_control_rates:{name}_rate'
                u_rate2_name = f'polynomial_control_rates:{name}_rate2'

                self._ivc.add_output(uhat_name, shape=(num_control_input_nodes,) + shape, units=units)
                self.add_design_var(uhat_name)
                self.add_constraint(u_name)
                self.add_constraint(u_rate_name)
                self.add_constraint(u_rate2_name)

                self.promotes('control_interp', inputs=[uhat_name],
                              outputs=[u_name, u_rate_name, u_rate2_name])

                # Promote targets from the ODE
                for tgt in targets:
                    self.promotes('ode', inputs=[(tgt, u_name)])
                if targets:
                    self.set_input_defaults(name=u_name,
                                            val=np.ones(shape),
                                            units=options['units'])
                # Promote rate targets from the ODE
                for tgt in rate_targets:
                    self.promotes('ode', inputs=[(tgt, u_rate_name)])
                if rate_targets:
                    self.set_input_defaults(name=u_rate_name,
                                            val=np.ones(shape),
                                            units=rate_units)

                # Promote rate2 targets from the ODE
                for tgt in rate2_targets:
                    self.promotes('ode', inputs=[(tgt, u_rate2_name)])
                if rate2_targets:
                    self.set_input_defaults(name=u_rate2_name,
                                            val=np.ones(shape),
                                            units=rate2_units)

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
        var = self._state_options[state_var]['rate_source']
        t_name = self._time_options['name']

        if var == t_name:
            rate_path = t_name
        elif var == f'{t_name}_phase':
            rate_path = f'{t_name}_phase'
        elif self._state_options is not None and var in self._state_options:
            rate_path = f'states:{var}'
        elif self._control_options is not None and var in self._control_options:
            rate_path = f'control_values:{var}'
        elif self._polynomial_control_options is not None and var in self._polynomial_control_options:
            rate_path = f'polynomial_control_values:{var}'
        elif self._parameter_options is not None and var in self._parameter_options:
            rate_path = f'parameters:{var}'
        elif var.endswith('_rate') and self._control_options is not None and \
                var[:-5] in self._control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate2') and self._control_options is not None and \
                var[:-6] in self._control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate') and self._polynomial_control_options is not None and \
                var[:-5] in self._polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        elif var.endswith('_rate2') and self._polynomial_control_options is not None and \
                var[:-6] in self._polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        else:
            rate_path = f'ode.{var}'
        return rate_path
