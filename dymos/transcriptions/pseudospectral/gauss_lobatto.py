from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from six import iteritems

from .pseudospectral_base import PseudospectralBase
from ..common import GaussLobattoPathConstraintComp, GaussLobattoTimeseriesOutputComp, \
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
    Dynamics 19.3 (1996): 592–599.
..
    """
    def setup_grid(self, phase):
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

    # def setup_design_parameters(self, phase):
    #     pass
    #
    # def setup_input_parameters(self, phase):
    #     pass
    #
    # def setup_traj_parameters(self, phase):
    #     super(GaussLobatto, self.setup)
    #
    # def setup_states(self, phase):
    #     super(GaussLobatto, self).setup_states(phase)

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
                phase.connect(
                     'states:{0}'.format(name),
                     ['rhs_disc.{0}'.format(tgt) for tgt in options['targets']],
                     src_indices=src_idxs, flat_src_indices=True)
                phase.connect(
                     'state_interp.state_col:{0}'.format(name),
                     ['rhs_col.{0}'.format(tgt) for tgt in options['targets']])

            rate_path, src_idxs = self.get_rate_source_path(name, nodes='state_disc', phase=phase)

            phase.connect(rate_path,
                          'state_interp.staterate_disc:{0}'.format(name),
                          src_indices=src_idxs)

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
            path_comp = GaussLobattoPathConstraintComp(grid_data=gd)
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
                src_idxs = get_src_indices_by_row(gd.input_maps['state_input_to_disc'], state_shape)
                phase.connect(src_name='states:{0}'.format(var),
                              tgt_name='path_constraints.disc_values:{0}'.format(con_name),
                              src_indices=src_idxs, flat_src_indices=True)
                phase.connect(src_name='state_interp.state_col:{0}'.format(var),
                              tgt_name='path_constraints.col_values:{0}'.format(con_name))

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
                    warnings.warn('Unable to infer shape of path constraint "%s" in phase "%s". Assuming scalar.\n'
                                  'In Dymos 1.0 the shape of ODE outputs must be explictly provided'
                                  ' via the add_path_constraint method.'%(var, phase.name), DeprecationWarning)
                    options['shape'] = (1,)
                phase.connect(src_name='rhs_disc.{0}'.format(var),
                              tgt_name='path_constraints.disc_values:{0}'.format(con_name))
                phase.connect(src_name='rhs_col.{0}'.format(var),
                              tgt_name='path_constraints.col_values:{0}'.format(con_name))

            kwargs = options.copy()
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data
        time_units = phase.time_options['units']
        timeseries_comp = GaussLobattoTimeseriesOutputComp(grid_data=gd)
        phase.add_subsystem('timeseries', subsys=timeseries_comp)

        timeseries_comp._add_timeseries_output('time',
                                               var_class=phase.classify_var('time'),
                                               units=time_units)
        phase.connect(src_name='time', tgt_name='timeseries.all_values:time')

        timeseries_comp._add_timeseries_output('time_phase',
                                               var_class=phase.classify_var('time_phase'),
                                               units=time_units)
        phase.connect(src_name='time_phase', tgt_name='timeseries.all_values:time_phase')

        for name, options in iteritems(phase.state_options):
            timeseries_comp._add_timeseries_output('states:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=options['units'])
            src_rows = gd.input_maps['state_input_to_disc']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            phase.connect(src_name='states:{0}'.format(name),
                          tgt_name='timeseries.disc_values:states:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)
            phase.connect(src_name='state_interp.state_col:{0}'.format(name),
                          tgt_name='timeseries.col_values:states:{0}'.format(name))

        for name, options in iteritems(phase.control_options):
            control_units = options['units']

            # Control values
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            phase.connect(src_name='control_values:{0}'.format(name),
                          tgt_name='timeseries.all_values:controls:{0}'.format(name))

            # # Control rates
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            phase.connect(src_name='control_rates:{0}_rate'.format(name),
                          tgt_name='timeseries.all_values:control_rates:{0}_rate'.format(name))

            # Control second derivatives
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            phase.connect(src_name='control_rates:{0}_rate2'.format(name),
                          tgt_name='timeseries.all_values:control_rates:{0}_rate2'.format(name))

        for name, options in iteritems(phase.polynomial_control_options):
            control_units = options['units']

            # Control values
            timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            phase.connect(src_name='polynomial_control_values:{0}'.format(name),
                          tgt_name='timeseries.all_values:'
                                   'polynomial_controls:{0}'.format(name))

            # # Control rates
            timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            phase.connect(src_name='polynomial_control_rates:{0}_rate'.format(name),
                          tgt_name='timeseries.all_values:'
                                   'polynomial_control_rates:{0}_rate'.format(name))

            # Control second derivatives
            timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                   '{0}_rate2'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            phase.connect(src_name='polynomial_control_rates:{0}_rate2'.format(name),
                          tgt_name='timeseries.all_values:'
                                   'polynomial_control_rates:{0}_rate2'.format(name))

        for name, options in iteritems(phase.design_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='design_parameters:{0}'.format(name),
                          tgt_name='timeseries.all_values:design_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.input_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='input_parameters:{0}_out'.format(name),
                          tgt_name='timeseries.all_values:input_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.traj_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('traj_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='traj_parameters:{0}_out'.format(name),
                          tgt_name='timeseries.all_values:traj_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for var, options in iteritems(phase._timeseries_outputs):
            output_name = options['output_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            # Failed to find variable, assume it is in the ODE
            phase.connect(src_name='rhs_disc.{0}'.format(var),
                          tgt_name='timeseries.disc_values:{0}'.format(output_name))
            phase.connect(src_name='rhs_col.{0}'.format(var),
                          tgt_name='timeseries.col_values:{0}'.format(output_name))

            kwargs = options.copy()
            kwargs.pop('output_name', None)
            timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

    def get_rate_source_path(self, state_name, nodes, phase):
        gd = self.grid_data
        try: 
            var = phase.state_options[state_name]['rate_source']
        except RuntimeError: 
            raise ValueError('state "%s" in phase "%s" was not given a rate_source'%(state_name, phase.name))
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
