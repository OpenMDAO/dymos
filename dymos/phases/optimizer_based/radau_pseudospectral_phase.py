from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from openmdao.utils.units import convert_units, valid_units
from six import iteritems

from ..grid_data import GridData, make_subset_map
from .optimizer_based_phase_base import OptimizerBasedPhaseBase
from ..components import RadauPathConstraintComp, RadauPSContinuityComp, RadauTimeseriesOutputComp
from ...utils.misc import get_rate_units
from ...utils.indexing import get_src_indices_by_row


class RadauPseudospectralPhase(OptimizerBasedPhaseBase):
    """
    RadauPseudospectralPhase implements Legendre-Gauss-Radau
    pseudospectral transcription for solving optimal control problems.

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

        super(RadauPseudospectralPhase, self).__init__(**kwgs)

        # Pluck out the kwargs needed to initialize grid_data, potentially needed prior to setup.
        num_segments = num_segments
        transcription_order = transcription_order
        segment_ends = segment_ends
        compressed = compressed
        self.grid_data = GridData(num_segments=num_segments, transcription='radau-ps',
                                  transcription_order=transcription_order,
                                  segment_ends=segment_ends, compressed=compressed)

    def initialize(self, **kwargs):
        super(RadauPseudospectralPhase, self).initialize(**kwargs)
        self.options['transcription'] = 'radau-ps'

    def _setup_time(self):
        comps = super(RadauPseudospectralPhase, self)._setup_time()

        if self.time_options['targets']:
            self.connect('time',
                         ['rhs_all.{0}'.format(t) for t in self.time_options['targets']],
                         src_indices=self.grid_data.subset_node_indices['all'])

        if self.time_options['time_phase_targets']:
            self.connect('time_phase',
                         ['rhs_all.{0}'.format(t) for t in self.time_options['time_phase_targets']],
                         src_indices=self.grid_data.subset_node_indices['all'])

        if self.time_options['t_initial_targets']:
            tgts = self.time_options['t_initial_targets']
            self.connect('t_initial',
                         ['rhs_all.{0}'.format(t) for t in tgts])

        if self.time_options['t_duration_targets']:
            tgts = self.time_options['t_duration_targets']
            self.connect('t_duration',
                         ['rhs_all.{0}'.format(t) for t in tgts])
        return comps

    def _setup_controls(self):
        super(RadauPseudospectralPhase, self)._setup_controls()

        for name, options in iteritems(self.control_options):

            if self.control_options[name]['targets']:
                targets = self.control_options[name]['targets']

                self.connect('control_values:{0}'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets])

            if self.control_options[name]['rate_targets']:
                targets = self.control_options[name]['rate_targets']
                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets])

            if self.control_options[name]['rate2_targets']:
                targets = self.control_options[name]['rate2_targets']
                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets])

    def _setup_polynomial_controls(self):
        super(RadauPseudospectralPhase, self)._setup_polynomial_controls()

        for name, options in iteritems(self.polynomial_control_options):

            if self.polynomial_control_options[name]['targets']:
                targets = self.polynomial_control_options[name]['targets']

                self.connect('polynomial_control_values:{0}'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets])

            if self.polynomial_control_options[name]['rate_targets']:
                targets = self.polynomial_control_options[name]['rate_targets']
                self.connect('polynomial_control_rates:{0}_rate'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets])

            if self.polynomial_control_options[name]['rate2_targets']:
                targets = self.polynomial_control_options[name]['rate2_targets']
                self.connect('polynomial_control_rates:{0}_rate2'.format(name),
                             ['rhs_all.{0}'.format(t) for t in targets])

    def _get_parameter_connections(self, name):
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

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        path_comp = None
        gd = self.grid_data
        time_units = self.time_options['units']

        if self._path_constraints:
            path_comp = RadauPathConstraintComp(grid_data=gd)
            self.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(self._path_constraints):
            constraint_kwargs = options.copy()
            con_units = constraint_kwargs['units'] = options.get('units', None)
            con_name = constraint_kwargs.pop('constraint_name')

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            if var_type == 'time':
                constraint_kwargs['shape'] = (1,)
                constraint_kwargs['units'] = time_units if con_units is None else con_units
                constraint_kwargs['linear'] = True
                self.connect(src_name='time',
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            if var_type == 'time_phase':
                constraint_kwargs['shape'] = (1,)
                constraint_kwargs['units'] = time_units if con_units is None else con_units
                constraint_kwargs['linear'] = True
                self.connect(src_name='time_phase',
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'state':
                state_shape = self.state_options[var]['shape']
                state_units = self.state_options[var]['units']
                constraint_kwargs['shape'] = state_shape
                constraint_kwargs['units'] = state_units if con_units is None else con_units
                constraint_kwargs['linear'] = False
                src_idxs = get_src_indices_by_row(gd.input_maps['state_input_to_disc'], state_shape)
                self.connect(src_name='states:{0}'.format(var),
                             tgt_name='path_constraints.all_values:{0}'.format(con_name),
                             src_indices=src_idxs, flat_src_indices=True)

            elif var_type == 'indep_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True
                constraint_path = 'control_values:{0}'.format(var)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'input_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']

                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = True
                constraint_path = 'control_values:{0}'.format(var)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'indep_polynomial_control':
                control_shape = self.polynomial_control_options[var]['shape']
                control_units = self.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False
                constraint_path = 'polynomial_control_values:{0}'.format(var)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'input_polynomial_control':
                control_shape = self.polynomial_control_options[var]['shape']
                control_units = self.polynomial_control_options[var]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = control_units if con_units is None else con_units
                constraint_kwargs['linear'] = False
                constraint_path = 'polynomial_control_values:{0}'.format(var)

                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                control_shape = self.polynomial_control_options[control_name]['shape']
                control_units = self.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=1) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                control_shape = self.polynomial_control_options[control_name]['shape']
                control_units = self.polynomial_control_options[control_name]['units']
                constraint_kwargs['shape'] = control_shape
                constraint_kwargs['units'] = get_rate_units(control_units, time_units, deriv=2) \
                    if con_units is None else con_units
                constraint_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the ODE
                constraint_kwargs['linear'] = False
                constraint_kwargs['shape'] = options.get('shape', None)
                if constraint_kwargs['shape'] is None:
                    warnings.warn('Unable to infer shape of path constraint {0}. Assuming scalar.\n'
                                  'In Dymos 1.0 the shape of ODE outputs must be explictly provided'
                                  ' via the add_path_constraint method.', DeprecationWarning)
                    constraint_kwargs['shape'] = (1,)
                self.connect(src_name='rhs_all.{0}'.format(var),
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            path_comp._add_path_constraint(con_name, var_type, **constraint_kwargs)

    def _setup_timeseries_outputs(self):

        gd = self.grid_data
        time_units = self.time_options['units']
        timeseries_comp = RadauTimeseriesOutputComp(grid_data=gd)
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
                         tgt_name='timeseries.all_values:states:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['all']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            self.connect(src_name='control_values:{0}'.format(name),
                         tgt_name='timeseries.all_values:controls:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

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
            timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['all']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            self.connect(src_name='polynomial_control_values:{0}'.format(name),
                         tgt_name='timeseries.all_values:'
                                  'polynomial_controls:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

            # # Control rates
            timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            self.connect(src_name='polynomial_control_rates:{0}_rate'.format(name),
                         tgt_name='timeseries.all_values:polynomial_control_rates:'
                                  '{0}_rate'.format(name))

            # Control second derivatives
            timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                   '{0}_rate2'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            self.connect(src_name='polynomial_control_rates:{0}_rate2'.format(name),
                         tgt_name='timeseries.all_values:polynomial_control_rates:'
                                  '{0}_rate2'.format(name))

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
                                                   units=units)

            if options['dynamic']:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
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
            self.connect(src_name='rhs_all.{0}'.format(var),
                         tgt_name='timeseries.all_values:{0}'.format(output_name))

            kwargs = options.copy()
            kwargs.pop('output_name', None)
            timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

    def _setup_rhs(self):
        super(RadauPseudospectralPhase, self)._setup_rhs()

        ODEClass = self.options['ode_class']
        grid_data = self.grid_data
        num_input_nodes = grid_data.subset_num_nodes['state_input']

        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']

        kwargs = self.options['ode_init_kwargs']
        self.add_subsystem('rhs_all',
                           subsys=ODEClass(num_nodes=grid_data.subset_num_nodes['all'],
                                           **kwargs))

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
                    ['rhs_all.{0}'.format(tgt) for tgt in options['targets']],
                    src_indices=src_idxs, flat_src_indices=True)

    def _setup_defects(self):
        super(RadauPseudospectralPhase, self)._setup_defects()
        grid_data = self.grid_data

        for name, options in iteritems(self.state_options):
            self.connect(
                'state_interp.staterate_col:{0}'.format(name),
                'collocation_constraint.f_approx:{0}'.format(name))

            rate_src, src_idxs = self._get_rate_source_path(name, nodes='col')

            self.connect(rate_src,
                         'collocation_constraint.f_computed:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        if grid_data.num_segments > 1:
            self.add_subsystem('continuity_comp',
                               RadauPSContinuityComp(grid_data=grid_data,
                                                     state_options=self.state_options,
                                                     control_options=self.control_options,
                                                     time_units=self.time_options['units']),
                               promotes_inputs=['t_duration'])

    def _get_rate_source_path(self, state_name, nodes, **kwargs):
        gd = self.grid_data
        var = self.state_options[state_name]['rate_source']
        # Note the rate source must be shape-compatible with the state
        shape = self.state_options[state_name]['shape']
        var_type = self._classify_var(var)

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
        elif var_type == 'design_parameter':
            rate_path = 'design_parameters:{0}'.format(var)
            dynamic = self.design_parameter_options[var]['dynamic']
            if dynamic:
                node_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
            else:
                node_idxs = np.zeros(1, dtype=int)
        elif var_type == 'input_parameter':
            rate_path = 'input_parameters:{0}_out'.format(var)
            dynamic = self.input_parameter_options[var]['dynamic']
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
