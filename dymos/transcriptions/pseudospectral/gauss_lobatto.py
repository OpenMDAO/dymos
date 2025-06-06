
import numpy as np
import openmdao.api as om

from .pseudospectral_base import PseudospectralBase
from .components import GaussLobattoInterleaveComp
from ..common import GaussLobattoContinuityComp
from ...utils.misc import get_rate_units
from ...utils.introspection import get_promoted_vars, get_targets, get_source_metadata
from ...utils.indexing import get_src_indices_by_row
from ...utils.ode_utils import _make_ode_system
from ..grid_data import GridData, make_subset_map


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
        num_segments = self.options['num_segments']

        if isinstance(self.options['order'], str):
            self.options['order'] = num_segments * [self.options['order']]
        elif np.ndim(self.options['order']) == 0:
            order = np.ones(num_segments, int) * self.options['order']
            self.options['order'] = np.asarray(order, dtype=int)
        else:
            self.options['order'] = np.asarray(self.options['order'], dtype=int)

        if np.any(self.options['order'] % 2 == 0):
            raise ValueError('A Gauss-Lobatto scheme must use an odd order for state interpolation.')

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
        for name, usr_tgts in [('t', options['targets']),
                               ('t_phase', options['time_phase_targets']),
                               ('dt_dstau', options['dt_dstau_targets'])]:

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

    def configure_timeseries_outputs(self, phase):
        """
        Configure the IO of the timeseries.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_timeseries_outputs(phase)
        self.configure_interleave_comp(phase)

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(GaussLobatto, self).configure_controls(phase)

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

            if options['targets']:
                phase.connect(f'control_values:{name}',
                              [f'rhs_disc.{t}' for t in options['targets']],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'control_values:{name}',
                              [f'rhs_col.{t}' for t in options['targets']],
                              src_indices=col_src_idxs, flat_src_indices=True)

            # Rate targets
            if options['rate_targets']:
                phase.connect(f'control_rates:{name}_rate',
                              [f'rhs_disc.{t}' for t in options['rate_targets']],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'control_rates:{name}_rate',
                              [f'rhs_col.{t}' for t in options['rate_targets']],
                              src_indices=col_src_idxs, flat_src_indices=True)

            # Second time derivative targets must be specified explicitly
            if options['rate2_targets']:
                phase.connect(f'control_rates:{name}_rate2',
                              [f'rhs_disc.{t}' for t in options['rate2_targets']],
                              src_indices=disc_src_idxs, flat_src_indices=True)

                phase.connect(f'control_rates:{name}_rate2',
                              [f'rhs_col.{t}' for t in options['rate2_targets']],
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

        rhs_disc = _make_ode_system(ode_class=ode_class,
                                    num_nodes=grid_data.subset_num_nodes['state_disc'],
                                    ode_init_kwargs=phase.options['ode_init_kwargs'],
                                    calc_exprs=phase._calc_exprs,
                                    parameter_options=phase.parameter_options)
        rhs_col = _make_ode_system(ode_class=ode_class,
                                   num_nodes=grid_data.subset_num_nodes['col'],
                                   ode_init_kwargs=phase.options['ode_init_kwargs'],
                                   calc_exprs=phase._calc_exprs,
                                   parameter_options=phase.parameter_options)

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
                              f'interleave_comp.col_values:states:{state_name}',
                              src_indices=om.slicer[...])

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

        for timeseries_name, timeseries_options in phase._timeseries.items():
            for ts_output_name, ts_output in timeseries_options['outputs'].items():
                name = ts_output['name']
                var_type = phase.classify_var(name)
                if var_type == 'ode':
                    units = ts_output['units']
                    shape = ts_output['shape']

                    # Add the state values to the interleave comp
                    src_added = interleave_comp.add_var(ts_output_name, shape, units,
                                                        disc_src=f'rhs_disc.{ts_output_name}',
                                                        col_src=f'rhs_col.{ts_output_name}')

                    if src_added:
                        phase.connect(f'rhs_disc.{ts_output["name"]}',
                                      f'interleave_comp.disc_values:{ts_output_name}',
                                      src_indices=om.slicer[...])

                        phase.connect(f'rhs_col.{ts_output["name"]}',
                                      f'interleave_comp.col_values:{ts_output_name}',
                                      src_indices=om.slicer[...])

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

        for name in phase.state_options:
            phase.connect(f'state_interp.staterate_col:{name}',
                          f'collocation_constraint.f_approx:{name}')

            rate_path, src_idxs = self._get_rate_source_path(name, nodes='col', phase=phase)

            phase.connect(rate_path,
                          f'collocation_constraint.f_computed:{name}',
                          src_indices=src_idxs)

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
            raise ValueError(f"state '{state_name}' in phase '{phase.name}' was not given a rate_source")
        var_type = phase.classify_var(var)

        # Determine the path to the variable
        if var_type == 't':
            rate_path = 't'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 't_phase':
            rate_path = 't_phase'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'state':
            if nodes == 'col':
                rate_path = f'state_interp.state_col:{var}'
                node_idxs = np.arange(gd.subset_num_nodes[nodes], dtype=int)
            elif nodes == 'state_disc':
                rate_path = f'states:{var}'
                node_idxs = make_subset_map(gd.subset_node_indices['state_input'],
                                            gd.subset_node_indices[nodes])
        elif var_type == 'control':
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
                raise ValueError(f'Unabled to find rate path for variable {var} at '
                                 f'node subset {nodes}')
        src_idxs = om.slicer[node_idxs, ...]

        return rate_path, src_idxs

    def _get_timeseries_var_source(self, var, output_name, phase):
        """
        Return the source path and indices for a given variable to be connected to a timeseries.

        Parameters
        ----------
        var : str
            Name of the timeseries variable whose source is desired.
        output_name : str
            Name of the timeseries output whose source is desired.
        phase : dymos.Phase
            Phase object containing the variable, either as state, time, control, etc., or as an ODE output.

        Returns
        -------
        meta : dict
            Metadata pertaining to the variable at the given path. This dict contains 'src' (the path to the
            timeseries source), 'src_idxs' (an array of the
            source indices), 'units' (the units of the source variable), and 'shape' (the shape of the variable at
            a given node).
        """
        gd = self.grid_data
        var_type = phase.classify_var(var)
        time_units = phase.time_options['units']

        transcription = phase.options['transcription']
        ode = transcription._get_ode(phase)
        ode_outputs = get_promoted_vars(ode, 'output')

        # The default for node_idxs, applies to everything except states and parameters.
        node_idxs = gd.subset_node_indices['all']
        meta = {}

        # Determine the path to the variable
        if var_type == 't':
            path = 't'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 't_phase':
            path = 't_phase'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 'state':
            path = f'interleave_comp.all_values:states:{var}'
            src_units = phase.state_options[var]['units']
            src_shape = phase.state_options[var]['shape']
        elif var_type == 'control':
            path = f'control_values:{var}'
            src_units = phase.control_options[var]['units']
            src_shape = phase.control_options[var]['shape']
        elif var_type == 'control_rate':
            control_name = var[:-5]
            path = f'control_rates:{control_name}_rate'
            control_name = var[:-5]
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=1)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            path = f'control_rates:{control_name}_rate2'
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=2)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'parameter':
            path = f'parameter_vals:{var}'
            node_idxs = np.zeros(gd.subset_num_nodes['all'], dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            path = f'interleave_comp.all_values:{output_name}'
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(f'ODE output {var} is tagged with "dymos.static_output" '
                                   'and cannot be a timeseries output.')

        src_idxs = om.slicer[node_idxs, ...]

        meta['src'] = path
        meta['src_idxs'] = src_idxs
        meta['units'] = src_units
        meta['shape'] = src_shape

        return meta

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
            # targets = options['targets']
            # static = options['static_targets']
            shape = options['shape']

            for tgt in options['targets']:
                if tgt in options['static_targets']:
                    inds = np.squeeze(get_src_indices_by_row([0], shape), axis=0)
                    disc_src_idxs = inds
                    col_src_idxs = inds
                else:
                    disc_rows = np.zeros(self.grid_data.subset_num_nodes['state_disc'], dtype=int)
                    col_rows = np.zeros(self.grid_data.subset_num_nodes['col'], dtype=int)
                    disc_src_idxs = get_src_indices_by_row(disc_rows, shape)
                    col_src_idxs = get_src_indices_by_row(col_rows, shape)
                    if shape == (1,):
                        disc_src_idxs = disc_src_idxs.ravel()
                        col_src_idxs = col_src_idxs.ravel()

                # enclose indices in tuple to ensure shaping of indices works
                disc_src_idxs = (disc_src_idxs,)
                col_src_idxs = (col_src_idxs,)

                connection_info.append((f'rhs_disc.{tgt}', disc_src_idxs))
                connection_info.append((f'rhs_col.{tgt}', col_src_idxs))

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
