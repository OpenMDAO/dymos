from collections import defaultdict
import warnings

import numpy as np
import openmdao.api as om
from openmdao.utils.general_utils import simple_warning

from .pseudospectral_base import PseudospectralBase
from .components import GaussLobattoInterleaveComp
from ..common import GaussLobattoContinuityComp
from ...utils.misc import get_rate_units, _unspecified
from ...utils.introspection import get_promoted_vars, get_targets, get_source_metadata
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData, make_subset_map
from fnmatch import filter


class GaussLobatto(PseudospectralBase):
    """
    High-order Gauss Lobatto Transcription.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    References
    ----------
    Herman, Albert L, and Bruce A Conway. "Direct Optimization Using Collocation Based on
    High-Order Gauss-Lobatto Quadrature Rules." Journal of Guidance, Control, and
    Dynamics 19.3 (1996): 592-599.
    """
    def __init__(self, **kwargs):
        super(GaussLobatto, self).__init__(**kwargs)
        self._rhs_source = 'rhs_disc'

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription='gauss-lobatto',
                                  transcription_order=self.options['order'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(GaussLobatto, self).configure_time(phase)
        options = phase.time_options
        ode_inputs = get_promoted_vars(self._get_ode(phase), 'input')

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, usr_tgts in [('time', options['targets']),
                               ('time_phase', options['time_phase_targets'])]:

            targets = get_targets(ode_inputs, name=name, user_targets=usr_tgts)
            if targets:
                disc_src_idxs = self.grid_data.subset_node_indices['state_disc']
                col_src_idxs = self.grid_data.subset_node_indices['col']
                phase.connect(name,
                              [f'rhs_col.{t}' for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)
                phase.connect(name,
                              [f'rhs_disc.{t}' for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

        for name, targets in [('t_initial', options['t_initial_targets']),
                              ('t_duration', options['t_duration_targets'])]:
            for t in targets:
                shape = ode_inputs[t]['shape']
                if shape == (1,):
                    disc_src_idxs = None
                    col_src_idxs = None
                    flat_src_idxs = None
                    src_shape = None
                else:
                    disc_src_idxs = self.grid_data.subset_node_indices['state_disc']
                    col_src_idxs = self.grid_data.subset_node_indices['col']
                    flat_src_idxs = True
                    src_shape = (1,)

                phase.promotes('rhs_disc', inputs=[(t, name)], src_indices=disc_src_idxs,
                               flat_src_indices=flat_src_idxs, src_shape=src_shape)
                phase.promotes('rhs_col', inputs=[(t, name)], src_indices=col_src_idxs,
                               flat_src_indices=flat_src_idxs, src_shape=src_shape)
            if targets:
                phase.set_input_defaults(name=name,
                                         val=np.ones((1,)),
                                         units=options['units'])

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(GaussLobatto, self).configure_controls(phase)
        ode_inputs = get_promoted_vars(self._get_ode(phase), 'input')

        grid_data = self.grid_data

        for name, options in phase.control_options.items():
            disc_idxs = grid_data.subset_node_indices['state_disc']
            col_idxs = grid_data.subset_node_indices['col']

            disc_src_idxs = get_src_indices_by_row(disc_idxs, options['shape'])
            col_src_idxs = get_src_indices_by_row(col_idxs, options['shape'])

            if options['shape'] == (1,):
                disc_src_idxs = disc_src_idxs.ravel()
                col_src_idxs = col_src_idxs.ravel()

            # enclose indices in tuple to ensure shaping of indices works
            disc_src_idxs = (disc_src_idxs,)
            col_src_idxs = (col_src_idxs,)

            # Control targets are detected automatically
            targets = get_targets(ode_inputs, name, options['targets'])

            if targets:
                phase.connect(f'control_values:{name}',
                              [f'rhs_disc.{t}' for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'control_values:{name}',
                              [f'rhs_col.{t}' for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            # Rate targets
            targets = get_targets(ode_inputs, name, options['rate_targets'], control_rates=1)

            if targets:
                phase.connect(f'control_rates:{name}_rate',
                              [f'rhs_disc.{t}' for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'control_rates:{name}_rate',
                              [f'rhs_col.{t}' for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            # Second time derivative targets must be specified explicitly
            targets = get_targets(ode_inputs, name, options['rate2_targets'], control_rates=2)
            if targets:
                phase.connect(f'control_rates:{name}_rate2',
                              [f'rhs_disc.{t}' for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'control_rates:{name}_rate2',
                              [f'rhs_col.{t}' for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(GaussLobatto, self).configure_polynomial_controls(phase)
        ode_inputs = get_promoted_vars(self._get_ode(phase), 'input')
        grid_data = self.grid_data

        for name, options in phase.polynomial_control_options.items():
            disc_idxs = grid_data.subset_node_indices['state_disc']
            col_idxs = grid_data.subset_node_indices['col']

            disc_src_idxs = get_src_indices_by_row(disc_idxs, options['shape'])
            col_src_idxs = get_src_indices_by_row(col_idxs, options['shape'])

            if options['shape'] == (1,):
                disc_src_idxs = disc_src_idxs.ravel()
                col_src_idxs = col_src_idxs.ravel()

            # enclose indices in tuple to ensure shaping of indices works
            disc_src_idxs = (disc_src_idxs,)
            col_src_idxs = (col_src_idxs,)

            targets = get_targets(ode=ode_inputs, name=name, user_targets=options['targets'])
            if targets:
                phase.connect(f'polynomial_control_values:{name}',
                              [f'rhs_disc.{t}' for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)
                phase.connect(f'polynomial_control_values:{name}',
                              [f'rhs_col.{t}' for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            targets = get_targets(ode=ode_inputs, name=name, user_targets=options['rate_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate',
                              [f'rhs_disc.{t}' for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'polynomial_control_rates:{name}_rate',
                              [f'rhs_col.{t}' for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

            targets = get_targets(ode=ode_inputs, name=name, user_targets=options['rate2_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate2',
                              [f'rhs_disc.{t}' for t in targets],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'polynomial_control_rates:{name}_rate2',
                              [f'rhs_col.{t}' for t in targets],
                              src_indices=col_src_idxs, flat_src_indices=True)

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data
        ode_class = phase.options['ode_class']

        kwargs = phase.options['ode_init_kwargs']
        rhs_disc = ode_class(num_nodes=grid_data.subset_num_nodes['state_disc'], **kwargs)
        rhs_col = ode_class(num_nodes=grid_data.subset_num_nodes['col'], **kwargs)

        phase.add_subsystem('rhs_disc', rhs_disc)

        super(GaussLobatto, self).setup_ode(phase)

        phase.add_subsystem('rhs_col', rhs_col)

        # Setup the interleave comp to interleave all states, any path constraints from the ODE,
        # and any timeseries outputs from the ODE.
        #
        phase.add_subsystem('interleave_comp', GaussLobattoInterleaveComp(grid_data=self.grid_data))

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(GaussLobatto, self).configure_ode(phase)

        ode_inputs = get_promoted_vars(self._get_ode(phase), 'input')
        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']

        for name, options in phase.state_options.items():
            src_idxs = om.slicer[map_input_indices_to_disc, ...]
            targets = get_targets(ode=ode_inputs, name=name, user_targets=options['targets'])

            if targets:
                phase.connect(f'states:{name}',
                              [f'rhs_disc.{tgt}' for tgt in targets],
                              src_indices=src_idxs)
                phase.connect(f'state_interp.state_col:{name}',
                              [f'rhs_col.{tgt}' for tgt in targets])

            rate_path, disc_src_idxs = self._get_rate_source_path(name, nodes='state_disc',
                                                                  phase=phase)
            phase.connect(rate_path,
                          f'state_interp.staterate_disc:{name}',
                          src_indices=disc_src_idxs, flat_src_indices=False)
        self.configure_interleave_comp(phase)

    def configure_interleave_comp(self, phase):
        """
        Create connections to the interleave_comp.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        interleave_comp = phase._get_subsystem('interleave_comp')
        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']

        time_units = phase.time_options['units']

        for state_name, options in phase.state_options.items():
            shape = options['shape']
            units = options['units']
            rate_src = options['rate_source']

            # Add the state values to the interleave comp
            src_added = interleave_comp.add_var(f'states:{state_name}', shape, units,
                                                disc_src=f'states:{state_name}',
                                                col_src=f'state_interp.state_col:{state_name}')

            if src_added:
                phase.connect(f'states:{state_name}',
                              f'interleave_comp.disc_values:states:{state_name}',
                              src_indices=om.slicer[map_input_indices_to_disc, ...])

                phase.connect(f'state_interp.state_col:{state_name}',
                              f'interleave_comp.col_values:states:{state_name}')

            # Add the state rates to the interleave comp

            if rate_src in phase.parameter_options:
                rate_path_disc = rate_path_col = f'parameters:{rate_src}'
            else:
                rate_path_disc, disc_src_idxs = self._get_rate_source_path(state_name,
                                                                           nodes='state_disc',
                                                                           phase=phase)
                rate_path_col, col_src_idxs = self._get_rate_source_path(state_name,
                                                                         nodes='col',
                                                                         phase=phase)

            src_added = interleave_comp.add_var(f'state_rates:{state_name}', shape,
                                                units=get_rate_units(options['units'], time_units),
                                                disc_src=rate_path_disc, col_src=rate_path_col)

            if src_added:
                rate_path_disc, disc_src_idxs = self._get_rate_source_path(state_name,
                                                                           nodes='state_disc',
                                                                           phase=phase)
                phase.connect(rate_path_disc,
                              f'interleave_comp.disc_values:state_rates:{state_name}',
                              src_indices=disc_src_idxs)

                rate_path_col, col_src_idxs = self._get_rate_source_path(state_name,
                                                                         nodes='col',
                                                                         phase=phase)
                phase.connect(rate_path_col,
                              f'interleave_comp.col_values:state_rates:{state_name}',
                              src_indices=col_src_idxs)

    def setup_defects(self, phase):
        """
        Create the continuity_comp to house the defects.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(GaussLobatto, self).setup_defects(phase)

        if any(self._requires_continuity_constraints(phase)):
            phase.add_subsystem('continuity_comp',
                                GaussLobattoContinuityComp(grid_data=self.grid_data,
                                                           state_options=phase.state_options,
                                                           control_options=phase.control_options,
                                                           time_units=phase.time_options['units']))

    def configure_defects(self, phase):
        """
        Configure the continuity_comp and connect the collocation constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(GaussLobatto, self).configure_defects(phase)

        any_state_cnty, any_control_cnty, any_control_rate_cnty = self._requires_continuity_constraints(phase)

        if any((any_state_cnty, any_control_cnty, any_control_rate_cnty)):
            phase._get_subsystem('continuity_comp').configure_io()

        for name, options in phase.state_options.items():
            phase.connect(f'state_interp.staterate_col:{name}',
                          f'collocation_constraint.f_approx:{name}')

            rate_path, src_idxs = self._get_rate_source_path(name, nodes='col', phase=phase)

            phase.connect(rate_path,
                          f'collocation_constraint.f_computed:{name}',
                          src_indices=src_idxs)

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ode_outputs = get_promoted_vars(self._get_ode(phase), 'output')

        for timeseries_name, timeseries_options in phase._timeseries.items():
            timeseries_comp = phase._get_subsystem(timeseries_name)
            time_units = phase.time_options['units']

            timeseries_comp._add_output_configure('time', units=time_units, shape=(1,), src='time')
            timeseries_comp._add_output_configure('time_phase', units=time_units, shape=(1,), src='time_phase')

            phase.connect(src_name='time', tgt_name=f'{timeseries_name}.input_values:time')
            phase.connect(src_name='time_phase', tgt_name=f'{timeseries_name}.input_values:time_phase')

            for state_name, options in phase.state_options.items():
                timeseries_comp._add_output_configure(f'states:{state_name}',
                                                      shape=options['shape'],
                                                      units=options['units'],
                                                      desc=options['desc'],
                                                      src=f'interleave_comp.all_values:states:{state_name}')

                timeseries_comp._add_output_configure(f'state_rates:{state_name}',
                                                      shape=options['shape'],
                                                      units=get_rate_units(options['units'], time_units),
                                                      desc=f'time-derivative of state {state_name}',
                                                      src=f'interleave_comp.all_values:state_rates:{state_name}')

                phase.connect(src_name=f'interleave_comp.all_values:states:{state_name}',
                              tgt_name=f'{timeseries_name}.input_values:states:{state_name}')

                phase.connect(src_name=f'interleave_comp.all_values:state_rates:{state_name}',
                              tgt_name=f'{timeseries_name}.input_values:state_rates:{state_name}')

            for control_name, options in phase.control_options.items():
                control_units = options['units']

                # Control values
                timeseries_comp._add_output_configure(f'controls:{control_name}',
                                                      shape=options['shape'],
                                                      units=control_units,
                                                      desc=options['desc'],
                                                      src=f'control_values:{control_name}')

                phase.connect(src_name=f'control_values:{control_name}',
                              tgt_name=f'{timeseries_name}.input_values:controls:{control_name}')

                # Control rates
                timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=1),
                                                      desc=f'first time-derivative of {control_name}',
                                                      src=f'control_rates:{control_name}_rate')

                phase.connect(src_name=f'control_rates:{control_name}_rate',
                              tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate')

                # Control second derivatives
                timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate2',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=2),
                                                      desc=f'second time-derivative of {control_name}',
                                                      src=f'control_rates:{control_name}_rate2')

                phase.connect(src_name=f'control_rates:{control_name}_rate2',
                              tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate2')

            for control_name, options in phase.polynomial_control_options.items():
                control_units = options['units']

                # Control values
                phase.connect(src_name=f'polynomial_control_values:{control_name}',
                              tgt_name=f'{timeseries_name}.input_values:'
                                       f'polynomial_controls:{control_name}')

                timeseries_comp._add_output_configure(f'polynomial_controls:{control_name}',
                                                      shape=options['shape'],
                                                      units=control_units,
                                                      desc=options['desc'],
                                                      src=f'polynomial_control_values:{control_name}')

                # Control rates
                phase.connect(src_name=f'polynomial_control_rates:{control_name}_rate',
                              tgt_name=f'{timeseries_name}.input_values:'
                                       f'polynomial_control_rates:{control_name}_rate')

                timeseries_comp._add_output_configure(f'polynomial_control_rates:{control_name}_rate',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=1),
                                                      desc=f'first time-derivative of {control_name}',
                                                      src=f'polynomial_control_rates:{control_name}_rate')

                # Control second derivatives
                phase.connect(src_name=f'polynomial_control_rates:{control_name}_rate2',
                              tgt_name=f'{timeseries_name}.input_values:'
                                       f'polynomial_control_rates:{control_name}_rate2')

                timeseries_comp._add_output_configure(f'polynomial_control_rates:{control_name}_rate2',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=2),
                                                      desc=f'second time-derivative of {control_name}',
                                                      src=f'polynomial_control_rates:{control_name}_rate2')

            for param_name, options in phase.parameter_options.items():
                if options['include_timeseries']:
                    prom_name = f'parameters:{param_name}'
                    tgt_name = f'input_values:parameters:{param_name}'

                    # Add output.
                    timeseries_comp = phase._get_subsystem(timeseries_name)
                    timeseries_comp._add_output_configure(prom_name,
                                                          desc='',
                                                          shape=options['shape'],
                                                          units=options['units'],
                                                          src=prom_name)

                    src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                    src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape']).ravel()

                    phase.connect(f'parameter_vals:{param_name}',
                                  f'{timeseries_name}.{tgt_name}',
                                  src_indices=src_idxs, flat_src_indices=True)
                    # phase.promotes(timeseries_name, inputs=[(tgt_name, prom_name)],
                    #                src_indices=(src_idxs,), flat_src_indices=True)

            for ts_output in phase._timeseries[timeseries_name]['outputs']:
                var = ts_output['name']
                output_name = ts_output['output_name']
                units = ts_output['units']
                wildcard_units = ts_output['wildcard_units']
                shape = ts_output['shape']

                if '*' in var:  # match outputs from the ODE
                    matches = filter(list(ode_outputs.keys()), var)

                    # A nested ODE can have multiple outputs at different levels that share
                    #   the same name.
                    # If the user does not use the output_name option to add_timeseries_output
                    #   to disambiguate the variables with the same name, only one of the
                    #   variables will be added. This code warns the user if that is happening.
                    # Find the duplicate timeseries names by looking at the last part of the names.
                    output_name_groups = defaultdict(list)
                    for v in matches:
                        output_name = v.split('.')[-1]
                        output_name_groups[output_name].append(v)

                    # If there are duplicates, warn the user
                    for output_name, var_list in output_name_groups.items():
                        if len(var_list) > 1:
                            var_list_as_string = ', '.join(var_list)
                            simple_warning(f"The timeseries variable name {output_name} is "
                                           f"duplicated in these variables: {var_list_as_string}. "
                                           "Disambiguate by using the add_timeseries_output "
                                           "output_name option.")
                else:
                    matches = [var]

                for v in matches:
                    if '*' in var:
                        output_name = v.split('.')[-1]
                        units = ode_outputs[v]['units']
                        # check for wildcard_units override of ODE units
                        if v in wildcard_units:
                            units = wildcard_units[v]

                    # Determine the path to the variable which we will be constraining
                    # This is more complicated for path constraints since, for instance,
                    # a single state variable has two sources which must be connected to
                    # the path component.
                    var_type = phase.classify_var(v)

                    # Ignore any variables that we've already added (states, times, controls, etc)
                    if var_type != 'ode':
                        continue

                    # If the full shape does not start with num_nodes, skip this variable.
                    if self.is_static_ode_output(v, ode_outputs, self.grid_data.subset_num_nodes['state_disc']):
                        warnings.warn(f'Cannot add ODE output {v} to the timeseries output. It is '
                                      f'sized such that its first dimension != num_nodes.')
                        continue

                    try:
                        shape, units = get_source_metadata(ode_outputs, src=v,
                                                           user_units=units,
                                                           user_shape=shape)
                    except ValueError:
                        raise ValueError(f'Timeseries output {v} is not a known variable in'
                                         f' the phase {phase.pathname} nor is it a known output of '
                                         f' the ODE.')

                    timeseries_input_added = timeseries_comp._add_output_configure(output_name, units, shape,
                                                                                   src=f'interleave_comp.'
                                                                                       f'all_values:{output_name}')

                    interleave_comp = phase._get_subsystem('interleave_comp')
                    src_added = interleave_comp.add_var(output_name, shape, units,
                                                        disc_src=f'rhs_disc.{v}',
                                                        col_src=f'rhs_col.{v}')
                    if src_added:
                        phase.connect(src_name=f'rhs_disc.{v}',
                                      tgt_name=f'interleave_comp.disc_values:{output_name}')
                        phase.connect(src_name=f'rhs_col.{v}',
                                      tgt_name=f'interleave_comp.col_values:{output_name}')

                    if timeseries_input_added:
                        phase.connect(src_name=f'interleave_comp.all_values:{output_name}',
                                      tgt_name=f'{timeseries_name}.input_values:{output_name}')

    def _get_rate_source_path(self, state_name, nodes, phase):
        """
        Return the rate source location and indices for a given state name.

        Parameters
        ----------
        state_name : str
            Name of the state.
        nodes : str
            One of ['col', 'state_disc'].
        phase : dymos.Phase
            Phase object containing the rate source.

        Returns
        -------
        str
            Path to the rate source.
        ndarray
            Array of source indices.
        """
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
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'time_phase':
            rate_path = 'time_phase'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'state':
            if nodes == 'col':
                rate_path = f'state_interp.state_col:{var}'
                node_idxs = np.arange(gd.subset_num_nodes[nodes], dtype=int)
            elif nodes == 'state_disc':
                rate_path = f'states:{var}'
                node_idxs = make_subset_map(gd.subset_node_indices['state_input'],
                                            gd.subset_node_indices[nodes])
        elif var_type == 'indep_control':
            rate_path = f'control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_control':
            rate_path = f'control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate':
            control_name = var[:-5]
            rate_path = f'control_rates:{control_name}_rate'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            rate_path = f'control_rates:{control_name}_rate2'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'indep_polynomial_control':
            rate_path = f'polynomial_control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'input_polynomial_control':
            rate_path = f'polynomial_control_values:{var}'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            rate_path = f'polynomial_control_rates:{control_name}_rate'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            rate_path = f'polynomial_control_rates:{control_name}_rate2'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'parameter':
            rate_path = f'parameter_vals:{var}'
            node_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
        # Failed to find variable, it must be an ODE output
        else:
            # Failed to find variable, assume it is in the RHS
            if nodes == 'col':
                rate_path = f'rhs_col.{var}'
                node_idxs = np.arange(gd.subset_num_nodes[nodes], dtype=int)
            elif nodes == 'state_disc':
                rate_path = f'rhs_disc.{var}'
                node_idxs = np.arange(gd.subset_num_nodes[nodes], dtype=int)
            else:
                raise ValueError('Unabled to find rate path for variable {0} at '
                                 'node subset {1}'.format(var, nodes))
        src_idxs = om.slicer[node_idxs, ...]

        return rate_path, src_idxs

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            The name of the parameter whose connection info is desired.
        phase : dymos.Phase
            The phase to which this transcription instance applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design/input/traj parameter is to be connected.
        """
        connection_info = []

        if name in phase.parameter_options:
            options = phase.parameter_options[name]
            targets = options['targets']
            static = options['static_target']
            shape = options['shape']

            if not static:
                disc_rows = np.zeros(self.grid_data.subset_num_nodes['state_disc'], dtype=int)
                col_rows = np.zeros(self.grid_data.subset_num_nodes['col'], dtype=int)
                disc_src_idxs = get_src_indices_by_row(disc_rows, shape)
                col_src_idxs = get_src_indices_by_row(col_rows, shape)
                if shape == (1,):
                    disc_src_idxs = disc_src_idxs.ravel()
                    col_src_idxs = col_src_idxs.ravel()
            else:
                inds = np.squeeze(get_src_indices_by_row([0], shape), axis=0)
                disc_src_idxs = inds
                col_src_idxs = inds

            # enclose indices in tuple to ensure shaping of indices works
            disc_src_idxs = (disc_src_idxs,)
            col_src_idxs = (col_src_idxs,)

            rhs_disc_tgts = [f'rhs_disc.{t}' for t in targets]
            connection_info.append((rhs_disc_tgts, disc_src_idxs))

            rhs_col_tgts = [f'rhs_col.{t}' for t in targets]
            connection_info.append((rhs_col_tgts, col_src_idxs))

        return connection_info

    def _requires_continuity_constraints(self, phase):
        """
        Tests whether state and/or control and/or control rate continuity are required.

        Parameters
        ----------
        phase : dymos.Phase
            The phase to which this transcription applies.

        Returns
        -------
        any_state_continuity : bool
            True if any state continuity is required to be enforced.
        any_control_continuity : bool
            True if any control value continuity is required to be enforced.
        any_control_rate_continuity : bool
            True if any control rate continuity is required to be enforced.
        """
        num_seg = self.grid_data.num_segments
        compressed = self.grid_data.compressed

        any_state_continuity = num_seg > 1 and not compressed
        any_control_continuity = any([opts['continuity'] for opts in phase.control_options.values()])
        any_control_continuity = any_control_continuity and num_seg > 1 and not compressed
        any_rate_continuity = any([opts['rate_continuity'] or opts['rate2_continuity']
                                   for opts in phase.control_options.values()])
        any_rate_continuity = any_rate_continuity and num_seg > 1

        return any_state_continuity, any_control_continuity, any_rate_continuity
