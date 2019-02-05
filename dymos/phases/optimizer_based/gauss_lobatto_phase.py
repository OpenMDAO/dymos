from __future__ import division, print_function, absolute_import

import warnings
from six import iteritems

import numpy as np

from openmdao.utils.units import convert_units, valid_units

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

        return comps

    def _setup_controls(self):
        num_dynamic = super(GaussLobattoPhase, self)._setup_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.control_options):
            state_disc_idxs = grid_data.subset_node_indices['state_disc']
            col_idxs = grid_data.subset_node_indices['col']

            control_src_name = 'control_interp_comp.control_values:{0}'.format(name)

            if name in self.ode_options._parameters:
                targets = self.ode_options._parameters[name]['targets']
                self.connect(control_src_name,
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect(control_src_name,
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

            if options['rate_param']:
                targets = self.ode_options._parameters[options['rate_param']]['targets']

                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

            if options['rate2_param']:
                targets = self.ode_options._parameters[options['rate2_param']]['targets']

                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

                self.connect('control_rates:{0}_rate2'.format(name),
                             ['rhs_col.{0}'.format(t) for t in targets],
                             src_indices=col_idxs)

        return num_dynamic

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
        map_indices_to_disc = np.zeros(self.grid_data.subset_num_nodes['state_disc'], dtype=int)
        map_indices_to_col = np.zeros(self.grid_data.subset_num_nodes['col'], dtype=int)

        if name in self.ode_options._parameters:
            targets = self.ode_options._parameters[name]['targets']

            rhs_disc_tgts = ['rhs_disc.{0}'.format(t) for t in targets]
            connection_info.append((rhs_disc_tgts, map_indices_to_disc))

            rhs_col_tgts = ['rhs_col.{0}'.format(t) for t in targets]
            connection_info.append((rhs_col_tgts, map_indices_to_col))

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
                self.connect(src_name='states:{0}'.format(var),
                             tgt_name='path_constraints.disc_values:{0}'.format(con_name),
                             src_indices=gd.input_maps['state_input_to_disc'])
                self.connect(src_name='state_interp.state_col:{0}'.format(var),
                             tgt_name='path_constraints.col_values:{0}'.format(con_name))

            elif var_type in ('indep_control', 'input_control'):
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True

                constraint_path = 'control_interp_comp.control_values:{0}'.format(var)

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

            else:
                # Failed to find variable, assume it is in the RHS
                options['linear'] = False
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
            self.connect(src_name='control_interp_comp.control_values:{0}'.format(name),
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

        for name, options in iteritems(self.design_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            if self.ode_options._parameters[name]['dynamic']:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
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

            if self.ode_options._parameters[name]['dynamic']:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='input_parameters:{0}_out'.format(name),
                         tgt_name='timeseries.all_values:input_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for var, options in iteritems(self._timeseries_outputs):
            output_units = options.get('units', None)
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
            rate_path = 'control_interp_comp.control_values:{0}'.format(var)
            src_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_control':
            rate_path = 'control_interp_comp.control_values:{0}'.format(var)
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

    def add_objective(self, name, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False):
        """
        Allows the user to add an objective in the phase.  If name is not a state,
        control, control rate, or 'time', then this is assumed to be the path of the variable
        to be constrained in the RHS.

        Parameters
        ----------
        name : str
            Name of the objective variable.  This should be one of 'time', a state or control
            variable, or the path to an output from the top level of the RHS.
        loc : str
            Where in the phase the objective is to be evaluated.  Valid
            options are 'initial' and 'final'.  The default is 'final'.
        index : int, optional
            If variable is an array at each point in time, this indicates which index is to be
            used as the objective, assuming C-ordered flattening.
        shape : int, optional
            The shape of the objective variable, at a point in time
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        """
        var_type = self._classify_var(name)

        # Determine the path to the variable
        if var_type == 'time':
            obj_path = 'time'
        elif var_type == 'time_phase':
            obj_path = 'time_phase'
        elif var_type == 'state':
            obj_path = 'states:{0}'.format(name)
        elif var_type == 'indep_control':
            obj_path = 'control_interp_comp.control_values:{0}'.format(name)
        elif var_type == 'input_control':
            obj_path = 'control_interp_comp.control_values:{0}'.format(name)
        elif var_type == 'control_rate':
            control_name = name[:-5]
            obj_path = 'control_rates:{0}_rate'.format(control_name)
        elif var_type == 'control_rate2':
            control_name = name[:-6]
            obj_path = 'control_rates:{0}_rate2'.format(control_name)
        elif var_type == 'design_parameter':
            obj_path = 'design_parameters:{0}'.format(name)
        elif var_type == 'input_parameter':
            obj_path = 'input_parameters:{0}_out'.format(name)
        else:
            # Failed to find variable, assume it is in the RHS
            obj_path = 'rhs_disc.{0}'.format(name)

        super(GaussLobattoPhase, self)._add_objective(obj_path, loc=loc, index=index, shape=shape,
                                                      ref=ref, ref0=ref0, adder=adder,
                                                      scaler=scaler,
                                                      parallel_deriv_color=parallel_deriv_color,
                                                      vectorize_derivs=vectorize_derivs)

    # def get_values(self, var, nodes=None, units=None):
    #     """
    #     Retrieve the values of the given variable at the given
    #     subset of nodes.
    #
    #     Parameters
    #     ----------
    #     var : str
    #         The variable whose values are to be returned.  This may be
    #         the name 'time', the name of a state, control, or parameter,
    #         or the path to a variable in the ODEFunction of the phase.
    #     nodes : str
    #         The name of the node subset or None (default).
    #     units : str
    #         The units in which the values should be expressed.  Must be compatible
    #         with the corresponding units inside the phase.
    #
    #     Returns
    #     -------
    #     ndarray
    #         An array of the values at the requested node subset.  The
    #         node index is the first dimension of the ndarray.
    #     """
    #     dep_txt = """
    #     Method get_values has been deprecated.  To retrieve a values of
    #     a variable as a timeseries, access the timeseries outputs of the
    #     phase via the standard OpenMDAO get_val method, e.g.:
    #
    #     prob.get_val('phase.timeseries.time')
    #     prob.get_val('phase.timeseries.states:x')
    #     prob.get_val('phase.timeseries.controls:theta')
    #
    #     """
    #     warnings.warn(dep_txt, DeprecationWarning)
    #
    #     if nodes is None:
    #         nodes = 'all'
    #
    #     gd = self.grid_data
    #     disc_node_idxs = gd.subset_node_indices['state_disc']
    #     col_node_idxs = gd.subset_node_indices['col']
    #
    #     var_type = self._classify_var(var)
    #
    #     op = dict(self.list_outputs(explicit=True, values=True, units=True, shape=True,
    #                                 out_stream=None))
    #
    #     if units is not None:
    #         if not valid_units(units):
    #             raise ValueError('Units {0} is not a valid units identifier'.format(units))
    #
    #     var_prefix = '{0}.'.format(self.pathname) if self.pathname else ''
    #
    #     path_map = {'time': 'time.{0}',
    #                 'time_phase': 'time.{0}',
    #                 'state': ('indep_states.states:{0}', 'state_interp.state_col:{0}'),
    #                 'indep_control': 'control_interp_comp.control_values:{0}',
    #                 'input_control': 'control_interp_comp.control_values:{0}',
    #                 'design_parameter': 'design_params.design_parameters:{0}',
    #                 'input_parameter': 'input_params.input_parameters:{0}_out',
    #                 'control_rate': 'control_interp_comp.control_rates:{0}',
    #                 'control_rate2': 'control_interp_comp.control_rates:{0}',
    #                 'ode': ('rhs_disc.{0}', 'rhs_col.{0}')}
    #
    #     if var_type == 'state':
    #         # State and RHS values need to be interleaved since disc and col values are not
    #         # available from the same output
    #         disc_path_fmt, col_path_fmt = path_map[var_type]
    #         disc_path = var_prefix + disc_path_fmt.format(var)
    #         col_path = var_prefix + col_path_fmt.format(var)
    #
    #         state_shape = op[disc_path]['shape'][1:]
    #         disc_units = op[disc_path]['units']
    #         disc_vals = op[disc_path]['value']
    #         col_units = op[col_path]['units']
    #         col_vals = op[col_path]['value']
    #
    #         # If units is none, use the units from the IndepVarComp
    #         if units is None:
    #             units = disc_units
    #
    #         output_value = np.zeros((gd.num_nodes,) + state_shape)
    #         output_value[disc_node_idxs, ...] = \
    #             convert_units(disc_vals[gd.input_maps['state_input_to_disc'], ...],
    #                           disc_units, units)
    #         output_value[col_node_idxs, ...] = convert_units(col_vals, col_units, units)
    #
    #     elif var_type in ('indep_control', 'input_control'):
    #         var_path = var_prefix + path_map[var_type].format(var)
    #         output_units = op[var_path]['units']
    #
    #         vals = op[var_path]['value']
    #         output_value = convert_units(vals, output_units, units)
    #
    #     elif var_type in ('design_parameter', 'input_parameter', 'traj_design_parameter',
    #                       'traj_input_parameter'):
    #         var_path = var_prefix + path_map[var_type].format(var)
    #         output_units = op[var_path]['units']
    #
    #         output_value = convert_units(op[var_path]['value'], output_units, units)
    #         output_value = np.repeat(output_value, gd.num_nodes, axis=0)
    #
    #     elif var_type == 'ode':
    #         rhs_disc_outputs = dict(self.rhs_disc.list_outputs(out_stream=None, values=True,
    #                                                            shape=True, units=True))
    #         rhs_col_outputs = dict(self.rhs_col.list_outputs(out_stream=None, values=True,
    #                                                          shape=True, units=True))
    #
    #         prom2abs_disc = self.rhs_disc._var_allprocs_prom2abs_list
    #         prom2abs_col = self.rhs_col._var_allprocs_prom2abs_list
    #
    #         # Is var in prom2abs_disc['output']?
    #         abs_path_disc = prom2abs_disc['output'][var][0]
    #         abs_path_col = prom2abs_col['output'][var][0]
    #
    #         shape = rhs_disc_outputs[abs_path_disc]['shape'][1:]
    #         disc_units = rhs_disc_outputs[abs_path_disc]['units']
    #         col_units = rhs_col_outputs[abs_path_col]['units']
    #
    #         output_value = np.zeros((gd.num_nodes,) + shape)
    #
    #         disc_vals = rhs_disc_outputs[abs_path_disc]['value']
    #         col_vals = rhs_col_outputs[abs_path_col]['value']
    #
    #         output_value[disc_node_idxs, ...] = convert_units(disc_vals, disc_units, units)
    #         output_value[col_node_idxs, ...] = convert_units(col_vals, col_units, units)
    #
    #     else:
    #         var_path = var_prefix + path_map[var_type].format(var)
    #         output_units = op[var_path]['units']
    #         output_value = convert_units(op[var_path]['value'], output_units, units)
    #
    #     # Always return a column vector
    #     if len(output_value.shape) == 1:
    #         output_value = np.reshape(output_value, (gd.num_nodes, 1))
    #
    #     return output_value[gd.subset_node_indices[nodes], ...]
