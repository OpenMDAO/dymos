from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

from six import iteritems

from .pseudospectral_base import PseudospectralBase
from ..common import PathConstraintComp, RadauPSContinuityComp, PseudospectralTimeseriesOutputComp
from ...utils.misc import get_rate_units
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

        if phase.time_options['targets']:
            phase.connect('time',
                          ['rhs_all.{0}'.format(t) for t in phase.time_options['targets']],
                          src_indices=self.grid_data.subset_node_indices['all'])

        if phase.time_options['time_phase_targets']:
            phase.connect('time_phase',
                          ['rhs_all.{0}'.format(t) for t in phase.time_options['time_phase_targets']],
                          src_indices=self.grid_data.subset_node_indices['all'])

        if phase.time_options['t_initial_targets']:
            tgts = phase.time_options['t_initial_targets']
            phase.connect('t_initial',
                          ['rhs_all.{0}'.format(t) for t in tgts])

        if phase.time_options['t_duration_targets']:
            tgts = phase.time_options['t_duration_targets']
            phase.connect('t_duration',
                          ['rhs_all.{0}'.format(t) for t in tgts])

    def setup_controls(self, phase):
        super(Radau, self).setup_controls(phase)

        for name, options in iteritems(phase.control_options):

            if phase.control_options[name]['targets']:
                targets = phase.control_options[name]['targets']

                phase.connect('control_values:{0}'.format(name),
                              ['rhs_all.{0}'.format(t) for t in targets])

            if phase.control_options[name]['rate_targets']:
                targets = phase.control_options[name]['rate_targets']
                phase.connect('control_rates:{0}_rate'.format(name),
                              ['rhs_all.{0}'.format(t) for t in targets])

            if phase.control_options[name]['rate2_targets']:
                targets = phase.control_options[name]['rate2_targets']
                phase.connect('control_rates:{0}_rate2'.format(name),
                              ['rhs_all.{0}'.format(t) for t in targets])

    def setup_polynomial_controls(self, phase):
        super(Radau, self).setup_polynomial_controls(phase)

        for name, options in iteritems(phase.polynomial_control_options):

            if phase.polynomial_control_options[name]['targets']:
                targets = phase.polynomial_control_options[name]['targets']

                phase.connect('polynomial_control_values:{0}'.format(name),
                              ['rhs_all.{0}'.format(t) for t in targets])

            if phase.polynomial_control_options[name]['rate_targets']:
                targets = phase.polynomial_control_options[name]['rate_targets']
                phase.connect('polynomial_control_rates:{0}_rate'.format(name),
                              ['rhs_all.{0}'.format(t) for t in targets])

            if phase.polynomial_control_options[name]['rate2_targets']:
                targets = phase.polynomial_control_options[name]['rate2_targets']
                phase.connect('polynomial_control_rates:{0}_rate2'.format(name),
                              ['rhs_all.{0}'.format(t) for t in targets])

    def setup_ode(self, phase):
        super(Radau, self).setup_ode(phase)

        ODEClass = phase.options['ode_class']
        grid_data = self.grid_data
        num_input_nodes = grid_data.subset_num_nodes['state_input']

        map_input_indices_to_disc = grid_data.input_maps['state_input_to_disc']

        kwargs = phase.options['ode_init_kwargs']
        phase.add_subsystem('rhs_all',
                            subsys=ODEClass(num_nodes=grid_data.subset_num_nodes['all'],
                                            **kwargs))

        for name, options in iteritems(phase.state_options):
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')

            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            if options['shape'] == (1,):
                """ Flat state variable is passed as 1D data."""
                src_idxs = src_idxs.ravel()

            if options['targets']:
                phase.connect('states:{0}'.format(name),
                              ['rhs_all.{0}'.format(tgt) for tgt in options['targets']],
                              src_indices=src_idxs, flat_src_indices=True)

    def setup_defects(self, phase):
        super(Radau, self).setup_defects(phase)
        grid_data = self.grid_data

        for name, options in iteritems(phase.state_options):
            phase.connect('state_interp.staterate_col:{0}'.format(name),
                          'collocation_constraint.f_approx:{0}'.format(name))

            rate_src, src_idxs = self.get_rate_source_path(name, 'col', phase)

            phase.connect(rate_src,
                          'collocation_constraint.f_computed:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        if grid_data.num_segments > 1:
            phase.add_subsystem('continuity_comp',
                                RadauPSContinuityComp(grid_data=grid_data,
                                                      state_options=phase.state_options,
                                                      control_options=phase.control_options,
                                                      time_units=phase.time_options['units']),
                                promotes_inputs=['t_duration'])

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

        for var, options in iteritems(phase._path_constraints):
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
                phase.connect(src_name='time',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'time_phase':
                constraint_kwargs['shape'] = (1,)
                constraint_kwargs['units'] = time_units if con_units is None else con_units
                constraint_kwargs['linear'] = True
                phase.connect(src_name='time_phase',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                state_units = phase.state_options[var]['units']
                constraint_kwargs['shape'] = state_shape
                constraint_kwargs['units'] = state_units if con_units is None else con_units
                constraint_kwargs['linear'] = False
                src_idxs = get_src_indices_by_row(gd.input_maps['state_input_to_disc'], state_shape)
                phase.connect(src_name='states:{0}'.format(var),
                              tgt_name='path_constraints.all_values:{0}'.format(con_name),
                              src_indices=src_idxs, flat_src_indices=True)

            elif var_type == 'indep_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True
                constraint_path = 'control_values:{0}'.format(var)

                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'input_control':
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True
                constraint_path = 'control_values:{0}'.format(var)

                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'indep_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False
                constraint_path = 'polynomial_control_values:{0}'.format(var)

                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'input_polynomial_control':
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False
                constraint_path = 'polynomial_control_values:{0}'.format(var)

                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the ODE
                constraint_kwargs['linear'] = False
                constraint_kwargs['shape'] = options.get('shape', None)
                if constraint_kwargs['shape'] is None:
                    options['shape'] = (1,)
                    constraint_kwargs['shape'] = (1,)

                phase.connect(src_name='rhs_all.{0}'.format(var),
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            path_comp._add_path_constraint(con_name, var_type, **constraint_kwargs)

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data
        time_units = phase.time_options['units']

        for name, options in iteritems(phase._timeseries):

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
            phase.connect(src_name='time', tgt_name='{0}.input_values:time'.format(name))

            timeseries_comp._add_timeseries_output('time_phase',
                                                   var_class=phase.classify_var('time_phase'),
                                                   units=time_units)
            phase.connect(src_name='time_phase', tgt_name='{0}.input_values:time_phase'.format(name))

            for state_name, options in iteritems(phase.state_options):
                timeseries_comp._add_timeseries_output('states:{0}'.format(state_name),
                                                       var_class=phase.classify_var(state_name),
                                                       shape=options['shape'],
                                                       units=options['units'])
                src_rows = gd.input_maps['state_input_to_disc']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])
                phase.connect(src_name='states:{0}'.format(state_name),
                              tgt_name='{0}.input_values:states:{1}'.format(name, state_name),
                              src_indices=src_idxs, flat_src_indices=True)

            for control_name, options in iteritems(phase.control_options):
                control_units = options['units']
                timeseries_comp._add_timeseries_output('controls:{0}'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=control_units)
                src_rows = gd.subset_node_indices['all']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])
                phase.connect(src_name='control_values:{0}'.format(control_name),
                              tgt_name='{0}.input_values:controls:{1}'.format(name, control_name),
                              src_indices=src_idxs, flat_src_indices=True)

                # # Control rates
                timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=1))
                phase.connect(src_name='control_rates:{0}_rate'.format(control_name),
                              tgt_name='{0}.input_values:control_rates:{1}_rate'.format(name, control_name))

                # Control second derivatives
                timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=2))
                phase.connect(src_name='control_rates:{0}_rate2'.format(control_name),
                              tgt_name='{0}.input_values:control_rates:{1}_rate2'.format(name, control_name))

            for control_name, options in iteritems(phase.polynomial_control_options):
                control_units = options['units']
                timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=control_units)
                src_rows = gd.subset_node_indices['all']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])
                phase.connect(src_name='polynomial_control_values:{0}'.format(control_name),
                              tgt_name='{0}.input_values:'
                                       'polynomial_controls:{1}'.format(name, control_name),
                              src_indices=src_idxs, flat_src_indices=True)

                # # Control rates
                timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=1))
                phase.connect(src_name='polynomial_control_rates:{0}_rate'.format(control_name),
                              tgt_name='{0}.input_values:polynomial_control_rates:'
                                       '{1}_rate'.format(name, control_name))

                # Control second derivatives
                timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                       '{0}_rate2'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=2))
                phase.connect(src_name='polynomial_control_rates:{0}_rate2'.format(control_name),
                              tgt_name='{0}.input_values:polynomial_control_rates:'
                                       '{1}_rate2'.format(name, control_name))

            for param_name, options in iteritems(phase.design_parameter_options):
                units = options['units']
                timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(param_name),
                                                       var_class=phase.classify_var(param_name),
                                                       shape=options['shape'],
                                                       units=units)

                src_idxs_raw = np.zeros(gd.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                phase.connect(src_name='design_parameters:{0}'.format(param_name),
                              tgt_name='{0}.input_values:design_parameters:{1}'.format(name, param_name),
                              src_indices=src_idxs, flat_src_indices=True)

            for param_name, options in iteritems(phase.input_parameter_options):
                units = options['units']
                timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(param_name),
                                                       var_class=phase.classify_var(param_name),
                                                       shape=options['shape'],
                                                       units=units)

                src_idxs_raw = np.zeros(gd.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                phase.connect(src_name='input_parameters:{0}_out'.format(param_name),
                              tgt_name='{0}.input_values:input_parameters:{1}'.format(name, param_name),
                              src_indices=src_idxs, flat_src_indices=True)

            for param_name, options in iteritems(phase.traj_parameter_options):
                units = options['units']
                timeseries_comp._add_timeseries_output('traj_parameters:{0}'.format(param_name),
                                                       var_class=phase.classify_var(param_name),
                                                       units=units)

                src_idxs_raw = np.zeros(gd.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                phase.connect(src_name='traj_parameters:{0}_out'.format(param_name),
                              tgt_name='{0}.input_values:traj_parameters:{1}'.format(name, param_name),
                              src_indices=src_idxs, flat_src_indices=True)

            for var, options in iteritems(phase._timeseries[name]['outputs']):
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

                kwargs = options.copy()
                kwargs.pop('output_name', None)
                timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

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
            if self.options['compressed']:
                rate_path = 'states:{0}'.format(var)
                node_idxs = np.arange(gd.subset_num_nodes['state_input'] - 1, dtype=int)
            else:
                rate_path = 'states:{0}'.format(var)
                node_idxs = gd.subset_node_indices[nodes]
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
        elif var_type == 'design_parameter':
            rate_path = 'design_parameters:{0}'.format(var)
            dynamic = phase.design_parameter_options[var]['dynamic']
            if dynamic:
                node_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
            else:
                node_idxs = np.zeros(1, dtype=int)
        elif var_type == 'input_parameter':
            rate_path = 'input_parameters:{0}_out'.format(var)
            dynamic = phase.input_parameter_options[var]['dynamic']
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

        parameter_options = phase.design_parameter_options.copy()
        parameter_options.update(phase.input_parameter_options)
        parameter_options.update(phase.traj_parameter_options)
        parameter_options.update(phase.control_options)

        if name in parameter_options:
            try:
                targets = parameter_options[name]['targets']
            except KeyError:
                raise KeyError('Could not find any ODE targets associated with parameter {0}.'.format(name))

            dynamic = parameter_options[name]['dynamic']
            shape = parameter_options[name]['shape']

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
