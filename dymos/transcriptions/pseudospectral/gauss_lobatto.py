from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from six import iteritems

from .pseudospectral_base import PseudospectralBase
from .components import GaussLobattoInterleaveComp
from ..common import PathConstraintComp, PseudospectralTimeseriesOutputComp, \
    GaussLobattoContinuityComp
from ...utils.misc import get_rate_units
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData, make_subset_map


class GaussLobatto(PseudospectralBase):
    """
    High-order Gauss Lobatto Transcription

    References
    ----------
    Herman, Albert L, and Bruce A Conway. "Direct Optimization Using Collocation Based on
    High-Order Gauss-Lobatto Quadrature Rules." Journal of Guidance, Control, and
    Dynamics 19.3 (1996): 592-599.
    """
    def init_grid(self):
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription='gauss-lobatto',
                                  transcription_order=self.options['order'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def setup_time(self, phase):
        super(GaussLobatto, self).setup_time(phase)

        if phase.time_options['targets']:
            tgts = phase.time_options['targets']
            phase.connect('time',
                          ['rhs_col.{0}'.format(t) for t in tgts],
                          src_indices=self.grid_data.subset_node_indices['col'])
            phase.connect('time',
                          ['rhs_disc.{0}'.format(t) for t in tgts],
                          src_indices=self.grid_data.subset_node_indices['state_disc'])

        if phase.time_options['time_phase_targets']:
            tgts = phase.time_options['time_phase_targets']
            phase.connect('time_phase',
                          ['rhs_col.{0}'.format(t) for t in tgts],
                          src_indices=self.grid_data.subset_node_indices['col'])
            phase.connect('time_phase',
                          ['rhs_disc.{0}'.format(t) for t in tgts],
                          src_indices=self.grid_data.subset_node_indices['state_disc'])

        if phase.time_options['t_initial_targets']:
            tgts = phase.time_options['t_initial_targets']
            phase.connect('t_initial',
                          ['rhs_col.{0}'.format(t) for t in tgts])
            phase.connect('t_initial',
                          ['rhs_disc.{0}'.format(t) for t in tgts])

        if phase.time_options['t_duration_targets']:
            tgts = phase.time_options['t_duration_targets']
            phase.connect('t_duration',
                          ['rhs_col.{0}'.format(t) for t in tgts])
            phase.connect('t_duration',
                          ['rhs_disc.{0}'.format(t) for t in tgts])

    def setup_controls(self, phase):
        super(GaussLobatto, self).setup_controls(phase)
        grid_data = self.grid_data

        for name, options in iteritems(phase.control_options):
            disc_idxs = grid_data.subset_node_indices['state_disc']
            col_idxs = grid_data.subset_node_indices['col']

            disc_src_idxs = get_src_indices_by_row(disc_idxs, options['shape'])
            col_src_idxs = get_src_indices_by_row(col_idxs, options['shape'])

            if options['shape'] == (1,):
                disc_src_idxs = disc_src_idxs.ravel()
                col_src_idxs = col_src_idxs.ravel()

            if phase.control_options[name]['targets']:
                targets = phase.control_options[name]['targets']

                phase.connect('control_values:{0}'.format(name),
                              ['rhs_disc.{0}'.format(t) for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect('control_values:{0}'.format(name),
                              ['rhs_col.{0}'.format(t) for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            if phase.control_options[name]['rate_targets']:
                targets = phase.control_options[name]['rate_targets']

                phase.connect('control_rates:{0}_rate'.format(name),
                              ['rhs_disc.{0}'.format(t) for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect('control_rates:{0}_rate'.format(name),
                              ['rhs_col.{0}'.format(t) for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            if phase.control_options[name]['rate2_targets']:
                targets = phase.control_options[name]['rate2_targets']

                phase.connect('control_rates:{0}_rate2'.format(name),
                              ['rhs_disc.{0}'.format(t) for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect('control_rates:{0}_rate2'.format(name),
                              ['rhs_col.{0}'.format(t) for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

    def setup_polynomial_controls(self, phase):
        super(GaussLobatto, self).setup_polynomial_controls(phase)
        grid_data = self.grid_data

        for name, options in iteritems(phase.polynomial_control_options):
            disc_idxs = grid_data.subset_node_indices['state_disc']
            col_idxs = grid_data.subset_node_indices['col']

            disc_src_idxs = get_src_indices_by_row(disc_idxs, options['shape'])
            col_src_idxs = get_src_indices_by_row(col_idxs, options['shape'])

            if options['shape'] == (1,):
                disc_src_idxs = disc_src_idxs.ravel()
                col_src_idxs = col_src_idxs.ravel()

            if phase.polynomial_control_options[name]['targets']:
                targets = phase.polynomial_control_options[name]['targets']

                phase.connect('polynomial_control_values:{0}'.format(name),
                              ['rhs_disc.{0}'.format(t) for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect('polynomial_control_values:{0}'.format(name),
                              ['rhs_col.{0}'.format(t) for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            if phase.polynomial_control_options[name]['rate_targets']:
                targets = phase.polynomial_control_options[name]['rate_targets']

                phase.connect('polynomial_control_rates:{0}_rate'.format(name),
                              ['rhs_disc.{0}'.format(t) for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect('polynomial_control_rates:{0}_rate'.format(name),
                              ['rhs_col.{0}'.format(t) for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            if phase.polynomial_control_options[name]['rate2_targets']:
                targets = phase.polynomial_control_options[name]['rate2_targets']

                phase.connect('polynomial_control_rates:{0}_rate2'.format(name),
                              ['rhs_disc.{0}'.format(t) for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect('polynomial_control_rates:{0}_rate2'.format(name),
                              ['rhs_col.{0}'.format(t) for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

    def setup_ode(self, phase):
        grid_data = self.grid_data
        ode_class = phase.options['ode_class']
        num_input_nodes = self.grid_data.subset_num_nodes['state_input']

        kwargs = phase.options['ode_init_kwargs']
        rhs_disc = ode_class(num_nodes=grid_data.subset_num_nodes['state_disc'], **kwargs)
        rhs_col = ode_class(num_nodes=grid_data.subset_num_nodes['col'], **kwargs)

        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']

        phase.add_subsystem('rhs_disc', rhs_disc)

        super(GaussLobatto, self).setup_ode(phase)

        phase.add_subsystem('rhs_col', rhs_col)

        for name, options in iteritems(phase.state_options):
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')
            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            if size == 1:
                """ Flat state variable is passed as 1D data."""
                src_idxs = src_idxs.ravel()

            if options['targets']:
                phase.connect('states:{0}'.format(name),
                              ['rhs_disc.{0}'.format(tgt) for tgt in options['targets']],
                              src_indices=src_idxs, flat_src_indices=True)
                phase.connect('state_interp.state_col:{0}'.format(name),
                              ['rhs_col.{0}'.format(tgt) for tgt in options['targets']])

            rate_path, src_idxs = self.get_rate_source_path(name, nodes='state_disc', phase=phase)

            phase.connect(rate_path,
                          'state_interp.staterate_disc:{0}'.format(name),
                          src_indices=src_idxs)

        #
        # Setup the interleave comp to interleave all states, any path constraints from the ODE,
        # and any timeseries outputs from the ODE.
        #
        self.setup_interleave_comp(phase)

    def setup_interleave_comp(self, phase):
        num_input_nodes = self.grid_data.subset_num_nodes['state_input']

        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']

        interleave_comp = GaussLobattoInterleaveComp(grid_data=self.grid_data)

        #
        # First do the states
        #
        for state_name, options in iteritems(phase.state_options):
            shape = options['shape']
            units = options['units']
            interleave_comp.add_var('states:{0}'.format(state_name), shape, units)

            size = np.prod(options['shape'])
            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')
            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            if size == 1:
                """ Flat state variable is passed as 1D data."""
                src_idxs = src_idxs.ravel()

            phase.connect('states:{0}'.format(state_name),
                          'interleave_comp.disc_values:states:{0}'.format(state_name),
                          src_indices=src_idxs, flat_src_indices=True)

            phase.connect('state_interp.state_col:{0}'.format(state_name),
                          'interleave_comp.col_values:states:{0}'.format(state_name))

        #
        # Do the path constraints
        #
        for var, options in iteritems(phase._path_constraints):

            var_type = phase.classify_var(var)

            # We only need to interleave state variables (covered above) and ODE outputs
            if var_type != 'ode':
                continue

            shape = (1,) if options['shape'] is None else options['shape']
            units = options['units']
            con_name = options['constraint_name']

            if con_name in interleave_comp.vars:
                continue

            interleave_comp.add_var(con_name, shape, units)

            phase.connect(src_name='rhs_disc.{0}'.format(var),
                          tgt_name='interleave_comp.disc_values:{0}'.format(con_name))
            phase.connect(src_name='rhs_col.{0}'.format(var),
                          tgt_name='interleave_comp.col_values:{0}'.format(con_name))

        #
        # Do the timeseries outputs
        #
        for timeseries_name, timeseries_options in iteritems(phase._timeseries):

            for var, options in iteritems(timeseries_options['outputs']):

                var_type = phase.classify_var(var)

                # We only need to interleave state variables (covered above) and ODE outputs
                if var_type != 'ode':
                    continue

                # Assume scalar shape here, but check config will warn that it's inferred.
                output_name = options['output_name']
                shape = (1,) if options['shape'] is None else options['shape']
                units = options['units']

                if output_name in interleave_comp.vars:
                    continue

                interleave_comp.add_var(output_name, shape, units)

                phase.connect(src_name='rhs_disc.{0}'.format(var),
                              tgt_name='interleave_comp.disc_values:{0}'.format(output_name))
                phase.connect(src_name='rhs_col.{0}'.format(var),
                              tgt_name='interleave_comp.col_values:{0}'.format(output_name))

        phase.add_subsystem('interleave_comp', interleave_comp)

    def setup_defects(self, phase):
        super(GaussLobatto, self).setup_defects(phase)
        grid_data = self.grid_data

        for name, options in iteritems(phase.state_options):
            phase.connect('state_interp.staterate_col:{0}'.format(name),
                          'collocation_constraint.f_approx:{0}'.format(name))

            rate_path, src_idxs = self.get_rate_source_path(name, nodes='col', phase=phase)
            phase.connect(rate_path,
                          'collocation_constraint.f_computed:{0}'.format(name),
                          src_indices=src_idxs)

        if grid_data.num_segments > 1:
            phase.add_subsystem('continuity_comp',
                                GaussLobattoContinuityComp(grid_data=grid_data,
                                                           state_options=phase.state_options,
                                                           control_options=phase.control_options,
                                                           time_units=phase.time_options['units']),
                                promotes_inputs=['t_duration'])

    def setup_path_constraints(self, phase):
        path_comp = None
        gd = self.grid_data
        time_units = phase.time_options['units']

        if phase._path_constraints:
            path_comp = PathConstraintComp(num_nodes=gd.num_nodes)
            phase.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(phase._path_constraints):
            con_units = options.get('units', None)
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                phase.connect(src_name='time',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'time_phase':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                phase.connect(src_name='time_phase',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                state_units = phase.state_options[var]['units']
                options['shape'] = state_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = False
                phase.connect(src_name='interleave_comp.all_values:states:{0}'.format(var),
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type in ('indep_control', 'input_control'):
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True

                constraint_path = 'control_values:{0}'.format(var)

                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = False

                constraint_path = 'polynomial_control_values:{0}'.format(var)

                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
                phase.connect(src_name=constraint_path,
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the ODE
                options['linear'] = False
                if options['shape'] is None:
                    options['shape'] = (1,)
                phase.connect(src_name='interleave_comp.all_values:{0}'.format(con_name),
                              tgt_name='path_constraints.all_values:{0}'.format(con_name))

            kwargs = options.copy()
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

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
                phase.connect(src_name='interleave_comp.all_values:states:{0}'.format(state_name),
                              tgt_name='{0}.input_values:states:{1}'.format(name, state_name))

            for control_name, options in iteritems(phase.control_options):
                control_units = options['units']

                # Control values
                timeseries_comp._add_timeseries_output('controls:{0}'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=control_units)
                phase.connect(src_name='control_values:{0}'.format(control_name),
                              tgt_name='{0}.input_values:controls:{1}'.format(name, control_name))

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

                # Control values
                timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=control_units)
                phase.connect(src_name='polynomial_control_values:{0}'.format(control_name),
                              tgt_name='{0}.input_values:'
                                       'polynomial_controls:{1}'.format(name, control_name))

                # # Control rates
                timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=1))
                phase.connect(src_name='polynomial_control_rates:{0}_rate'.format(control_name),
                              tgt_name='{0}.input_values:'
                                       'polynomial_control_rates:{1}_rate'.format(name, control_name))

                # Control second derivatives
                timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                       '{0}_rate2'.format(control_name),
                                                       var_class=phase.classify_var(control_name),
                                                       shape=options['shape'],
                                                       units=get_rate_units(control_units,
                                                                            time_units,
                                                                            deriv=2))
                phase.connect(src_name='polynomial_control_rates:{0}_rate2'.format(control_name),
                              tgt_name='{0}.input_values:'
                                       'polynomial_control_rates:{1}_rate2'.format(name, control_name))

            for param_name, options in iteritems(phase.design_parameter_options):
                units = options['units']
                timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(param_name),
                                                       var_class=phase.classify_var(param_name),
                                                       shape=options['shape'],
                                                       units=units)

                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
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

                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                phase.connect(src_name='input_parameters:{0}_out'.format(param_name),
                              tgt_name='{0}.input_values:input_parameters:{1}'.format(name, param_name),
                              src_indices=src_idxs, flat_src_indices=True)

            for param_name, options in iteritems(phase.traj_parameter_options):
                units = options['units']
                timeseries_comp._add_timeseries_output('traj_parameters:{0}'.format(param_name),
                                                       var_class=phase.classify_var(param_name),
                                                       shape=options['shape'],
                                                       units=units)

                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
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

                # Assume scalar shape here, but check config will warn that it's inferred.
                if options['shape'] is None:
                    options['shape'] = (1,)

                # Failed to find variable, assume it is in the ODE
                phase.connect(src_name='interleave_comp.all_values:{0}'.format(output_name),
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
        var_type = phase.classify_var(var)

        # Determine the path to the variable
        if var_type == 'time':
            rate_path = 'time'
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'time_phase':
            rate_path = 'time_phase'
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'state':
            if nodes == 'col':
                rate_path = 'state_interp.state_col:{0}'.format(var)
                src_idxs = None
            elif nodes == 'state_disc':
                rate_path = 'states:{0}'.format(var)
                src_idxs = make_subset_map(gd.subset_node_indices['state_input'],
                                           gd.subset_node_indices[nodes])
        elif var_type == 'indep_control':
            rate_path = 'control_values:{0}'.format(var)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_control':
            rate_path = 'control_values:{0}'.format(var)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate':
            control_name = var[:-5]
            rate_path = 'control_rates:{0}_rate'.format(control_name)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            rate_path = 'control_rates:{0}_rate2'.format(control_name)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'indep_polynomial_control':
            rate_path = 'polynomial_control_values:{0}'.format(var)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_polynomial_control':
            rate_path = 'polynomial_control_values:{0}'.format(var)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            rate_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            rate_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'design_parameter':
            rate_path = 'design_parameters:{0}'.format(var)
            src_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
        elif var_type == 'input_parameter':
            rate_path = 'input_parameters:{0}_out'.format(var)
            src_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
        # Failed to find variable, it must be an ODE output
        else:
            # Failed to find variable, assume it is in the RHS
            if nodes == 'col':
                rate_path = 'rhs_col.{0}'.format(var)
                src_idxs = None
            elif nodes == 'state_disc':
                rate_path = 'rhs_disc.{0}'.format(var)
                src_idxs = None
            else:
                raise ValueError('Unabled to find rate path for variable {0} at '
                                 'node subset {1}'.format(var, nodes))

        return rate_path, src_idxs

    def get_parameter_connections(self, name, phase):
        """
        Returns a list containing tuples of each path and related indices to which the
        given design variable name is to be connected.

        Parameters
        ----------
        name : str
            The name of the parameter whose connection info is desired.
        phase
            The phase to which this transcription instance applies.

        Returns
        -------
        connection_info : list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design/input/traj parameter is to be connected.
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
                disc_rows = np.zeros(self.grid_data.subset_num_nodes['state_disc'], dtype=int)
                col_rows = np.zeros(self.grid_data.subset_num_nodes['col'], dtype=int)
                disc_src_idxs = get_src_indices_by_row(disc_rows, shape)
                col_src_idxs = get_src_indices_by_row(col_rows, shape)
                if shape == (1,):
                    disc_src_idxs = disc_src_idxs.ravel()
                    col_src_idxs = col_src_idxs.ravel()
            else:
                disc_src_idxs = np.squeeze(get_src_indices_by_row([0], shape), axis=0)
                col_src_idxs = np.squeeze(get_src_indices_by_row([0], shape), axis=0)

            rhs_disc_tgts = ['rhs_disc.{0}'.format(t) for t in targets]
            connection_info.append((rhs_disc_tgts, disc_src_idxs))

            rhs_col_tgts = ['rhs_col.{0}'.format(t) for t in targets]
            connection_info.append((rhs_col_tgts, col_src_idxs))

        return connection_info
