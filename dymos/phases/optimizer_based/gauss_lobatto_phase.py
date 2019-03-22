from __future__ import division, print_function, absolute_import

import warnings
from six import iteritems

import numpy as np

from ..grid_data import GridData, make_subset_map
from .optimizer_based_phase_base import OptimizerBasedPhaseBase
from ..components import GaussLobattoPathConstraintComp, GaussLobattoContinuityComp, \
    GaussLobattoTimeseriesOutputComp
from ...utils.misc import get_rate_units
from ...utils.indexing import get_src_indices_by_row


class GaussLobattoPhase(OptimizerBasedPhaseBase):
    """
    GaussLobattoPhase implements GaussLobatto transcription
    for solving optimal control problems.

    Parameters
    ----------
    num_segments : int
        The number of segments in the Phase
    transcription_order : int
        Order of transcription of the state variables within each segment.
    segment_ends : Iterable or None
        Iterable of locations of the segment ends in unnormalized space.  If None, segments
        will be equally distributed in the phase.
    compressed : bool
        If True, "compress" the transcription but providing only a single, shared value for
        states and controls at segment boundaries.
    **kwargs
        Additional options to be sent to the phase initialization as options.

    Attributes
    ----------
    self.time_options : dict of TimeOptionsDictionary
        A dictionary of options for time (integration variable) in the phase.

    self.state_options : dict of StateOptionsDictionary
        A dictionary of options for the RHS states in the Phase.

    self.control_options : dict of ControlOptionsDictionary
        A dictionary of options for the controls in the Phase.

    self._ode_controls : dict of ControlOptionsDictionary
        A dictionary of the default options for controllable inputs of the Phase RHS

    """

    def __init__(self, num_segments, transcription_order=3, segment_ends=None, compressed=True,
                 **kwargs):
        kwgs = kwargs.copy()
        kwgs.update({'num_segments': num_segments, 'transcription_order': transcription_order,
                    'segment_ends': segment_ends, 'compressed': compressed})

        super(GaussLobattoPhase, self).__init__(**kwgs)

        # Pluck out the kwargs needed to initialize grid_data, potentially needed prior to setup.
        num_segments = num_segments
        transcription_order = transcription_order
        segment_ends = segment_ends
        compressed = compressed
        self.grid_data = GridData(num_segments=num_segments, transcription='gauss-lobatto',
                                  transcription_order=transcription_order,
                                  segment_ends=segment_ends, compressed=compressed)

    def initialize(self, **kwargs):
        super(GaussLobattoPhase, self).initialize(**kwargs)
        self.options['transcription'] = 'gauss-lobatto'

    def _setup_time(self):
        comps = super(GaussLobattoPhase, self)._setup_time()

        if self.time_options['targets']:
            tgts = self.time_options['targets']
            self.connect('time',
                         ['rhs_col.{0}'.format(t) for t in tgts],
                         src_indices=self.grid_data.subset_node_indices['col'])
            self.connect('time',
                         ['rhs_disc.{0}'.format(t) for t in tgts],
                         src_indices=self.grid_data.subset_node_indices['state_disc'])

        if self.time_options['time_phase_targets']:
            tgts = self.time_options['time_phase_targets']
            self.connect('time_phase',
                         ['rhs_col.{0}'.format(t) for t in tgts],
                         src_indices=self.grid_data.subset_node_indices['col'])
            self.connect('time_phase',
                         ['rhs_disc.{0}'.format(t) for t in tgts],
                         src_indices=self.grid_data.subset_node_indices['state_disc'])

        if self.time_options['t_initial_targets']:
            tgts = self.time_options['t_initial_targets']
            self.connect('t_initial',
                         ['rhs_col.{0}'.format(t) for t in tgts])
            self.connect('t_initial',
                         ['rhs_disc.{0}'.format(t) for t in tgts])

        if self.time_options['t_duration_targets']:
            tgts = self.time_options['t_duration_targets']
            self.connect('t_duration',
                         ['rhs_col.{0}'.format(t) for t in tgts])
            self.connect('t_duration',
                         ['rhs_disc.{0}'.format(t) for t in tgts])

        return comps

    def _setup_controls(self):
        super(GaussLobattoPhase, self)._setup_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.control_options):
            state_disc_idxs = grid_data.subset_node_indices['state_disc']
            col_idxs = grid_data.subset_node_indices['col']

            if self.control_options[name]['targets']:
                targets = self.control_options[name]['targets']

                self.connect('control_values:{0}'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('control_values:{0}'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

            if self.control_options[name]['rate_targets']:
                targets = self.control_options[name]['rate_targets']

                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

            if self.control_options[name]['rate2_targets']:
                targets = self.control_options[name]['rate2_targets']

                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

    def _setup_polynomial_controls(self):
        super(GaussLobattoPhase, self)._setup_polynomial_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.polynomial_control_options):
            state_disc_idxs = grid_data.subset_node_indices['state_disc']
            col_idxs = grid_data.subset_node_indices['col']

            if self.polynomial_control_options[name]['targets']:
                targets = self.polynomial_control_options[name]['targets']

                self.connect('polynomial_control_values:{0}'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('polynomial_control_values:{0}'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

            if self.polynomial_control_options[name]['rate_targets']:
                targets = self.polynomial_control_options[name]['rate_targets']

                self.connect('polynomial_control_rates:{0}_rate'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('polynomial_control_rates:{0}_rate'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

            if self.polynomial_control_options[name]['rate2_targets']:
                targets = self.polynomial_control_options[name]['rate2_targets']

                self.connect('polynomial_control_rates:{0}_rate2'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('polynomial_control_rates:{0}_rate2'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

    def _get_parameter_connections(self, name):
        """
        Returns a list containing tuples of each path and related indices to which the
        given design variable name is to be connected.

        Returns
        -------
        connection_info : list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design/input/traj parameter is to be connected.
        """
        connection_info = []

        parameter_options = self.design_parameter_options.copy()
        parameter_options.update(self.input_parameter_options)
        parameter_options.update(self.traj_parameter_options)
        parameter_options.update(self.control_options)

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

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        path_comp = None
        gd = self.grid_data
        time_units = self.time_options['units']

        if self._path_constraints:
            path_comp = GaussLobattoPathConstraintComp(grid_data=gd)
            self.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(self._path_constraints):
            con_units = options.get('units', None)
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                self.connect(src_name='time',
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'time_phase':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                self.connect(src_name='time_phase',
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'state':
                state_shape = self.state_options[var]['shape']
                state_units = self.state_options[var]['units']
                options['shape'] = state_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = False
                src_idxs = get_src_indices_by_row(gd.input_maps['state_input_to_disc'], state_shape)
                self.connect(src_name='states:{0}'.format(var),
                             tgt_name='path_constraints.disc_values:{0}'.format(con_name),
                             src_indices=src_idxs, flat_src_indices=True)
                self.connect(src_name='state_interp.state_col:{0}'.format(var),
                             tgt_name='path_constraints.col_values:{0}'.format(con_name))

            elif var_type in ('indep_control', 'input_control'):
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True

                constraint_path = 'control_values:{0}'.format(var)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
                control_shape = self.polynomial_control_options[var]['shape']
                control_units = self.polynomial_control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = False

                constraint_path = 'polynomial_control_values:{0}'.format(var)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                control_shape = self.polynomial_control_options[control_name]['shape']
                control_units = self.polynomial_control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                control_shape = self.polynomial_control_options[control_name]['shape']
                control_units = self.polynomial_control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the ODE
                options['linear'] = False
                if options['shape'] is None:
                    warnings.warn('Unable to infer shape of path constraint {0}. Assuming scalar.\n'
                                  'In Dymos 1.0 the shape of ODE outputs must be explictly provided'
                                  ' via the add_path_constraint method.', DeprecationWarning)
                    options['shape'] = (1,)
                self.connect(src_name='rhs_disc.{0}'.format(var),
                             tgt_name='path_constraints.disc_values:{0}'.format(con_name))
                self.connect(src_name='rhs_col.{0}'.format(var),
                             tgt_name='path_constraints.col_values:{0}'.format(con_name))

            kwargs = options.copy()
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def _setup_timeseries_outputs(self):

        gd = self.grid_data
        time_units = self.time_options['units']
        timeseries_comp = GaussLobattoTimeseriesOutputComp(grid_data=gd)
        self.add_subsystem('timeseries', subsys=timeseries_comp)

        timeseries_comp._add_timeseries_output('time',
                                               var_class=self._classify_var('time'),
                                               units=time_units)
        self.connect(src_name='time', tgt_name='timeseries.all_values:time')

        timeseries_comp._add_timeseries_output('time_phase',
                                               var_class=self._classify_var('time_phase'),
                                               units=time_units)
        self.connect(src_name='time_phase', tgt_name='timeseries.all_values:time_phase')

        for name, options in iteritems(self.state_options):
            timeseries_comp._add_timeseries_output('states:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=options['units'])
            src_rows = gd.input_maps['state_input_to_disc']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            self.connect(src_name='states:{0}'.format(name),
                         tgt_name='timeseries.disc_values:states:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)
            self.connect(src_name='state_interp.state_col:{0}'.format(name),
                         tgt_name='timeseries.col_values:states:{0}'.format(name))

        for name, options in iteritems(self.control_options):
            control_units = options['units']

            # Control values
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            self.connect(src_name='control_values:{0}'.format(name),
                         tgt_name='timeseries.all_values:controls:{0}'.format(name))

            # # Control rates
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            self.connect(src_name='control_rates:{0}_rate'.format(name),
                         tgt_name='timeseries.all_values:control_rates:{0}_rate'.format(name))

            # Control second derivatives
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            self.connect(src_name='control_rates:{0}_rate2'.format(name),
                         tgt_name='timeseries.all_values:control_rates:{0}_rate2'.format(name))

        for name, options in iteritems(self.polynomial_control_options):
            control_units = options['units']

            # Control values
            timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            self.connect(src_name='polynomial_control_values:{0}'.format(name),
                         tgt_name='timeseries.all_values:'
                                  'polynomial_controls:{0}'.format(name))

            # # Control rates
            timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            self.connect(src_name='polynomial_control_rates:{0}_rate'.format(name),
                         tgt_name='timeseries.all_values:'
                                  'polynomial_control_rates:{0}_rate'.format(name))

            # Control second derivatives
            timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                   '{0}_rate2'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            self.connect(src_name='polynomial_control_rates:{0}_rate2'.format(name),
                         tgt_name='timeseries.all_values:'
                                  'polynomial_control_rates:{0}_rate2'.format(name))

        for name, options in iteritems(self.design_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='design_parameters:{0}'.format(name),
                         tgt_name='timeseries.all_values:design_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.input_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='input_parameters:{0}_out'.format(name),
                         tgt_name='timeseries.all_values:input_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.traj_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('traj_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='traj_parameters:{0}_out'.format(name),
                         tgt_name='timeseries.all_values:traj_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for var, options in iteritems(self._timeseries_outputs):
            output_name = options['output_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            # Failed to find variable, assume it is in the ODE
            self.connect(src_name='rhs_disc.{0}'.format(var),
                         tgt_name='timeseries.disc_values:{0}'.format(output_name))
            self.connect(src_name='rhs_col.{0}'.format(var),
                         tgt_name='timeseries.col_values:{0}'.format(output_name))

            kwargs = options.copy()
            kwargs.pop('output_name', None)
            timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

    def _get_rate_source_path(self, state_name, nodes, **kwargs):
        gd = self.grid_data
        var = self.state_options[state_name]['rate_source']
        var_type = self._classify_var(var)

        # Determine the path to the variable
        if var_type == 'time':
            rate_path = 'time'
            src_idxs = gd.subset_node_indices[nodes]
        if var_type == 'time_phase':
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
        else:
            # Failed to find variable, assume it is in the RHS
            if nodes == 'col':
                rate_path = 'rhs_col.{0}'.format(var)
                src_idxs = None
            elif nodes == 'state_disc':
                rate_path = 'rhs_disc.{0}'.format(var)
                src_idxs = None

        return rate_path, src_idxs

    def _setup_rhs(self):
        super(GaussLobattoPhase, self)._setup_rhs()

        grid_data = self.grid_data
        ODEClass = self.options['ode_class']
        num_input_nodes = self.grid_data.subset_num_nodes['state_input']

        kwargs = self.options['ode_init_kwargs']
        rhs_disc = ODEClass(num_nodes=grid_data.subset_num_nodes['state_disc'], **kwargs)
        rhs_col = ODEClass(num_nodes=grid_data.subset_num_nodes['col'], **kwargs)

        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']

        self.add_subsystem('rhs_disc', rhs_disc)
        self.add_subsystem('rhs_col', rhs_col)

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')
            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            if size == 1:
                """ Flat state variable is passed as 1D data."""
                src_idxs = src_idxs.ravel()

            if options['targets']:
                self.connect(
                    'states:{0}'.format(name),
                    ['rhs_disc.{0}'.format(tgt) for tgt in options['targets']],
                    src_indices=src_idxs, flat_src_indices=True)
                self.connect(
                    'state_interp.state_col:{0}'.format(name),
                    ['rhs_col.{0}'.format(tgt) for tgt in options['targets']])

            rate_path, src_idxs = self._get_rate_source_path(name, nodes='state_disc')

            self.connect(rate_path,
                         'state_interp.staterate_disc:{0}'.format(name),
                         src_indices=src_idxs)

    def _setup_defects(self):
        super(GaussLobattoPhase, self)._setup_defects()
        grid_data = self.grid_data

        for name, options in iteritems(self.state_options):
            self.connect(
                'state_interp.staterate_col:%s' % name,
                'collocation_constraint.f_approx:%s' % name)

            rate_path, src_idxs = self._get_rate_source_path(name, nodes='col')
            self.connect(rate_path,
                         'collocation_constraint.f_computed:%s' % name,
                         src_indices=src_idxs)

        if grid_data.num_segments > 1:
            self.add_subsystem('continuity_comp',
                               GaussLobattoContinuityComp(grid_data=grid_data,
                                                          state_options=self.state_options,
                                                          control_options=self.control_options,
                                                          time_units=self.time_options['units']),
                               promotes_inputs=['t_duration'])
