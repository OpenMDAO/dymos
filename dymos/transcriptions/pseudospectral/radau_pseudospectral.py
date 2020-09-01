import numpy as np

from .pseudospectral_base import PseudospectralBase
from ..common import PathConstraintComp, RadauPSContinuityComp, PseudospectralTimeseriesOutputComp
from ...utils.misc import get_rate_units, get_targets
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData


class Radau(PseudospectralBase):
    """
    Radau Pseudospectral Method Transcription

    References
    ----------
    Garg, Divya et al. "Direct Trajectory Optimization and Costate Estimation of General Optimal
    Control Problems Using a Radau Pseudospectral Method." American Institute of Aeronautics
    and Astronautics, 2009.
    """
    def init_grid(self):
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription='radau-ps',
                                  transcription_order=self.options['order'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def setup_time(self, phase):
        super(Radau, self).setup_time(phase)

    def configure_time(self, phase):
        options = phase.time_options

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, usr_tgts, dynamic in [('time', options['targets'], True),
                                        ('time_phase', options['time_phase_targets'], True),
                                        ('t_initial', options['t_initial_targets'], False),
                                        ('t_duration', options['t_duration_targets'], False)]:

            targets = get_targets(phase.rhs_all, name=name, user_targets=usr_tgts)
            if targets:
                src_idxs = self.grid_data.subset_node_indices['all'] if dynamic else None
                phase.connect(name, [f'rhs_all.{t}' for t in targets], src_indices=src_idxs)

    def configure_controls(self, phase):
        super(Radau, self).configure_controls(phase)

        if phase.control_options:
            for name, options in phase.control_options.items():
                targets = get_targets(ode=phase.rhs_all, name=name,
                                      user_targets=options['targets'])
                if targets:
                    phase.connect(f'control_values:{name}',
                                  [f'rhs_all.{t}' for t in targets])

                targets = get_targets(ode=phase.rhs_all, name=f'{name}_rate',
                                      user_targets=options['rate_targets'])
                if targets:
                    phase.connect(f'control_rates:{name}_rate',
                                  [f'rhs_all.{t}' for t in targets])

                targets = get_targets(ode=phase.rhs_all, name=f'{name}_rate2',
                                      user_targets=options['rate2_targets'])
                if targets:
                    phase.connect(f'control_rates:{name}_rate2',
                                  [f'rhs_all.{t}' for t in targets])

    def configure_polynomial_controls(self, phase):
        super(Radau, self).configure_polynomial_controls(phase)

        for name, options in phase.polynomial_control_options.items():
            targets = get_targets(ode=phase.rhs_all, name=name, user_targets=options['targets'])
            if targets:
                phase.connect(f'polynomial_control_values:{name}',
                              [f'rhs_all.{t}' for t in targets])

            targets = get_targets(ode=phase.rhs_all, name=f'{name}_rate',
                                  user_targets=options['rate_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate',
                              [f'rhs_all.{t}' for t in targets])

            targets = get_targets(ode=phase.rhs_all, name=f'{name}_rate2',
                                  user_targets=options['rate2_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate2',
                              [f'rhs_all.{t}' for t in targets])

    def setup_ode(self, phase):
        super(Radau, self).setup_ode(phase)

        ODEClass = phase.options['ode_class']
        grid_data = self.grid_data

        kwargs = phase.options['ode_init_kwargs']
        phase.add_subsystem('rhs_all',
                            subsys=ODEClass(num_nodes=grid_data.subset_num_nodes['all'],
                                            **kwargs))

    def configure_ode(self, phase):
        super(Radau, self).configure_ode(phase)

        grid_data = self.grid_data
        num_input_nodes = grid_data.subset_num_nodes['state_input']
        map_input_indices_to_disc = grid_data.input_maps['state_input_to_disc']

        for name, options in phase.state_options.items():
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')

            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            if options['shape'] == (1,):
                """ Flat state variable is passed as 1D data."""
                src_idxs = src_idxs.ravel()

            targets = get_targets(ode=phase.rhs_all, name=name, user_targets=options['targets'])
            if targets:
                phase.connect('states:{0}'.format(name),
                              ['rhs_all.{0}'.format(tgt) for tgt in targets],
                              src_indices=src_idxs, flat_src_indices=True)

    def setup_defects(self, phase):
        super(Radau, self).setup_defects(phase)

        grid_data = self.grid_data
        if grid_data.num_segments > 1:
            phase.add_subsystem('continuity_comp',
                                RadauPSContinuityComp(grid_data=grid_data,
                                                      state_options=phase.state_options,
                                                      control_options=phase.control_options,
                                                      time_units=phase.time_options['units']),
                                promotes_inputs=['t_duration'])

    def configure_defects(self, phase):
        for name, options in phase.state_options.items():
            phase.connect('state_interp.staterate_col:{0}'.format(name),
                          'collocation_constraint.f_approx:{0}'.format(name))

            rate_src, src_idxs = self.get_rate_source_path(name, 'col', phase)

            phase.connect(rate_src,
                          'collocation_constraint.f_computed:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

    def setup_path_constraints(self, phase):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        path_comp = None
        gd = self.grid_data
        time_units = phase.time_options['units']

        if phase._path_constraints:
            path_comp = PathConstraintComp(num_nodes=gd.num_nodes)
            phase.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in phase._path_constraints.items():
            constraint_kwargs = options.copy()
            con_units = constraint_kwargs['units'] = options.get('units', None)
            con_name = constraint_kwargs.pop('constraint_name')

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'time':
                constraint_kwargs['shape'] = (1,)
                constraint_kwargs['units'] = time_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'time_phase':
                constraint_kwargs['shape'] = (1,)
                constraint_kwargs['units'] = time_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                state_units = phase.state_options[var]['units']
                constraint_kwargs['shape'] = state_shape
                constraint_kwargs['units'] = state_units if con_units is None else con_units
                constraint_kwargs['linear'] = False

            elif var_type == 'indep_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'input_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True

            elif var_type == 'indep_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False

            elif var_type == 'input_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units

            else:
                # Failed to find variable, assume it is in the ODE
                constraint_kwargs['linear'] = False
                constraint_kwargs['shape'] = options.get('shape', None)
                if constraint_kwargs['shape'] is None:
                    options['shape'] = (1,)
                    constraint_kwargs['shape'] = (1,)

            path_comp._add_path_constraint(con_name, var_type, **constraint_kwargs)

    def configure_path_constraints(self, phase):
        gd = self.grid_data

        for var, options in phase._path_constraints.items():
            constraint_kwargs = options.copy()
            con_name = constraint_kwargs.pop('constraint_name')

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'time':
                phase.connect(src_name='time',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'time_phase':
                phase.connect(src_name='time_phase',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                src_idxs = get_src_indices_by_row(gd.input_maps['state_input_to_disc'], state_shape)
                phase.connect(src_name='states:{0}'.format(var),
                              tgt_name='path_constraints.all_values:{0}'.format(con_name),
                              src_indices=src_idxs, flat_src_indices=True)

            elif var_type == 'indep_control':
                constraint_path = 'control_values:{0}'.format(var)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'input_control':
                constraint_path = 'control_values:{0}'.format(var)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'indep_polynomial_control':
                constraint_path = 'polynomial_control_values:{0}'.format(var)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'input_polynomial_control':
                constraint_path = 'polynomial_control_values:{0}'.format(var)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate':
                control_name = var[:-5]
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                constraint_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                constraint_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the ODE
                phase.connect(src_name='rhs_all.{0}'.format(var),
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data
        time_units = phase.time_options['units']

        for name, options in phase._timeseries.items():
            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = PseudospectralTimeseriesOutputComp(input_grid_data=gd,
                                                                 output_grid_data=ogd,
                                                                 output_subset=options['subset'])
            phase.add_subsystem(name, subsys=timeseries_comp)

            timeseries_comp._add_timeseries_output('time',
                                                   var_class=phase.classify_var('time'),
                                                   units=time_units)

            timeseries_comp._add_timeseries_output('time_phase',
                                                   var_class=phase.classify_var('time_phase'),
                                                   units=time_units)

            for state_name, options in phase.state_options.items():
                timeseries_comp._add_timeseries_output('states:{0}'.format(state_name),
                                                       var_class=phase.classify_var(state_name),
                                                       shape=options['shape'],
                                                       units=options['units'])

                timeseries_comp._add_timeseries_output('state_rates:{0}'.format(state_name),
                                                       var_class=phase.classify_var(options['rate_source']),
                                                       shape=options['shape'],
                                                       units=get_rate_units(options['units'], time_units))

            for control_name, options in phase.control_options.items():
                control_units = options['units']
                timeseries_comp._add_timeseries_output('controls:{0}'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=control_units)

                # Control rates
                timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=1))

                # Control second derivatives
                timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=2))

            for control_name, options in phase.polynomial_control_options.items():
                control_units = options['units']
                timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=control_units)

                # Control rates
                timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=1))

                # Control second derivatives
                timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                       '{0}_rate2'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=2))

            # Parameters are delayed until configure so that we can query the units.

            for var, options in phase._timeseries[name]['outputs'].items():
                output_name = options['output_name']

                # Determine the path to the variable which we will be constraining
                # This is more complicated for path constraints since, for instance,
                # a single state variable has two sources which must be connected to
                # the path component.
                var_type = phase.classify_var(var)

                # Ignore any variables that we've already added (states, times, controls, etc)
                if var_type != 'ode':
                    continue

                # Assume scalar shape here if None, but check config will warn that it's inferred.
                if options['shape'] is None:
                    options['shape'] = (1,)

                kwargs = options.copy()
                kwargs.pop('output_name', None)
                timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

    def configure_timeseries_outputs(self, phase):
        gd = self.grid_data

        for name, options in phase._timeseries.items():
            phase.connect(src_name='time', tgt_name='{0}.input_values:time'.format(name))

            phase.connect(src_name='time_phase', tgt_name='{0}.input_values:time_phase'.format(name))

            for state_name, options in phase.state_options.items():
                src_rows = gd.input_maps['state_input_to_disc']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])
                phase.connect(src_name='states:{0}'.format(state_name),
                              tgt_name='{0}.input_values:states:{1}'.format(name, state_name),
                              src_indices=src_idxs, flat_src_indices=True)

                rate_src, src_idxs = self.get_rate_source_path(state_name, 'all', phase)
                phase.connect(src_name=rate_src,
                              tgt_name='{0}.input_values:state_rates:{1}'.format(name, state_name),
                              src_indices=src_idxs, flat_src_indices=True)

            for control_name, options in phase.control_options.items():
                src_rows = gd.subset_node_indices['all']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])
                phase.connect(src_name='control_values:{0}'.format(control_name),
                              tgt_name='{0}.input_values:controls:{1}'.format(name, control_name),
                              src_indices=src_idxs, flat_src_indices=True)

                # Control rates
                phase.connect(src_name='control_rates:{0}_rate'.format(control_name),
                              tgt_name='{0}.input_values:control_rates:{1}_rate'.format(name, control_name))

                # Control second derivatives
                phase.connect(src_name='control_rates:{0}_rate2'.format(control_name),
                              tgt_name='{0}.input_values:control_rates:{1}_rate2'.format(name, control_name))

            for control_name, options in phase.polynomial_control_options.items():
                src_rows = gd.subset_node_indices['all']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])
                phase.connect(src_name='polynomial_control_values:{0}'.format(control_name),
                              tgt_name='{0}.input_values:'
                                       'polynomial_controls:{1}'.format(name, control_name),
                              src_indices=src_idxs, flat_src_indices=True)

                # Control rates
                phase.connect(src_name='polynomial_control_rates:{0}_rate'.format(control_name),
                              tgt_name='{0}.input_values:polynomial_control_rates:'
                                       '{1}_rate'.format(name, control_name))

                # Control second derivatives
                phase.connect(src_name='polynomial_control_rates:{0}_rate2'.format(control_name),
                              tgt_name='{0}.input_values:polynomial_control_rates:'
                                       '{1}_rate2'.format(name, control_name))

            for param_name, options in phase.parameter_options.items():
                if options['include_timeseries']:
                    prom_name = 'parameters:{0}'.format(param_name)
                    tgt_name = 'input_values:parameters:{0}'.format(param_name)

                    targets = get_targets(phase.rhs_all, name=param_name, user_targets=options['targets'])

                    if targets:
                        prom_param = targets[0]
                    else:
                        prom_param = param_name

                    # Get the param's real units.
                    abs_param = phase.rhs_all._var_allprocs_prom2abs_list['input'][prom_param]
                    units = phase.rhs_all._var_abs2meta[abs_param[0]]['units']

                    # Add output.
                    timeseries_comp = phase._get_subsystem(name)
                    timeseries_comp._add_output_configure(prom_name,
                                                          desc='',
                                                          shape=options['shape'],
                                                          units=units)

                    src_idxs_raw = np.zeros(gd.subset_num_nodes['all'], dtype=int)
                    src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                    phase.promotes(name, inputs=[(tgt_name, prom_name)],
                                   src_indices=src_idxs, flat_src_indices=True)

            for var, options in phase._timeseries[name]['outputs'].items():
                output_name = options['output_name']

                # Determine the path to the variable which we will be constraining
                # This is more complicated for path constraints since, for instance,
                # a single state variable has two sources which must be connected to
                # the path component.
                var_type = phase.classify_var(var)

                # Ignore any variables that we've already added (states, times, controls, etc)
                if var_type != 'ode':
                    continue

                # Assume scalar shape here if None, but check config will warn that it's inferred.
                if options['shape'] is None:
                    options['shape'] = (1,)

                # Failed to find variable, assume it is in the ODE
                phase.connect(src_name='rhs_all.{0}'.format(var),
                              tgt_name='{0}.input_values:{1}'.format(name, output_name))

    def get_rate_source_path(self, state_name, nodes, phase):
        gd = self.grid_data
        try:
            var = phase.state_options[state_name]['rate_source']
        except RuntimeError:
            raise ValueError('state \'{0}\' in phase \'{1}\' was not given a '
                             'rate_source'.format(state_name, phase.name))

        # Note the rate source must be shape-compatible with the state
        shape = phase.state_options[state_name]['shape']
        var_type = phase.classify_var(var)

        # Determine the path to the variable
        if var_type == 'time':
            rate_path = 'time'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'time_phase':
            rate_path = 'time_phase'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'state':
            rate_path = 'states:{0}'.format(var)
            # Find the state_input indices which occur at segment endpoints, and repeat them twice
            state_input_idxs = gd.subset_node_indices['state_input']
            repeat_idxs = np.ones_like(state_input_idxs)
            if self.options['compressed']:
                segment_end_idxs = gd.subset_node_indices['segment_ends'][1:-1]
                # Repeat nodes that are on segment bounds (but not the first or last nodes in the phase)
                nodes_to_repeat = list(set(state_input_idxs).intersection(set(segment_end_idxs)))
                # Now find these nodes in the state input indices
                idxs_of_ntr_in_state_inputs = np.where(np.in1d(state_input_idxs, nodes_to_repeat))[0]
                # All state input nodes are used once, but nodes_to_repeat are used twice
                repeat_idxs[idxs_of_ntr_in_state_inputs] = 2
            # Now we have a way of mapping the state input indices to all nodes
            map_input_node_idxs_to_all = np.repeat(np.arange(gd.subset_num_nodes['state_input'],
                                                             dtype=int), repeats=repeat_idxs)
            # Now select the subset of nodes we want to use.
            node_idxs = map_input_node_idxs_to_all[gd.subset_node_indices[nodes]]
        elif var_type == 'indep_control':
            rate_path = 'control_values:{0}'.format(var)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_control':
            rate_path = 'control_values:{0}'.format(var)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate':
            control_name = var[:-5]
            rate_path = 'control_rates:{0}_rate'.format(control_name)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            rate_path = 'control_rates:{0}_rate2'.format(control_name)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'indep_polynomial_control':
            rate_path = 'polynomial_control_values:{0}'.format(var)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_polynomial_control':
            rate_path = 'polynomial_control_values:{0}'.format(var)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            rate_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            rate_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'parameter':
            rate_path = 'parameters:{0}'.format(var)
            dynamic = phase.parameter_options[var]['dynamic']
            if dynamic:
                node_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
            else:
                node_idxs = np.zeros(1, dtype=int)
        else:
            # Failed to find variable, assume it is in the ODE
            rate_path = 'rhs_all.{0}'.format(var)
            node_idxs = gd.subset_node_indices[nodes]

        src_idxs = get_src_indices_by_row(node_idxs, shape=shape)

        return rate_path, src_idxs

    def get_parameter_connections(self, name, phase):
        """
        Returns a list containing tuples of each path and related indices to which the
        given design variable name is to be connected.

        Returns
        -------
        connection_info : list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        connection_info = []

        if name in phase.parameter_options:
            options = phase.parameter_options[name]
            targets = get_targets(ode=phase.rhs_all, name=name, user_targets=options['targets'])

            dynamic = options['dynamic']
            shape = options['shape']

            if dynamic:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                if shape == (1,):
                    src_idxs = src_idxs.ravel()
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                src_idxs = np.squeeze(src_idxs, axis=0)

            rhs_all_tgts = ['rhs_all.{0}'.format(t) for t in targets]
            connection_info.append((rhs_all_tgts, src_idxs))

        return connection_info
