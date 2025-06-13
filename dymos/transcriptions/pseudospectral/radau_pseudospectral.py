import numpy as np
import openmdao.api as om

from .pseudospectral_base import PseudospectralBase
from ..common import RadauPSContinuityComp
from ...utils.misc import get_rate_units
from ...utils.introspection import get_promoted_vars, get_targets, get_source_metadata
from ...utils.indexing import get_src_indices_by_row
from ...utils.ode_utils import _make_ode_system
from ..grid_data import GridData


class Radau(PseudospectralBase):
    """
    Radau Pseudospectral Method Transcription.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    References
    ----------
    Garg, Divya et al. "Direct Trajectory Optimization and Costate Estimation of General Optimal
    Control Problems Using a Radau Pseudospectral Method." American Institute of Aeronautics
    and Astronautics, 2009.
    """
    def __init__(self, **kwargs):
        super(Radau, self).__init__(**kwargs)
        self._rhs_source = 'rhs_all'

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription='radau-ps',
                                  transcription_order=self.options['order'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        This method assumes that target introspection has already been performed by the phase and thus
        options['targets'], options['time_phase_targets'], options['t_initial_targets'],
        and options['t_duration_targets'] are all correctly populated.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).configure_time(phase)
        options = phase.time_options
        ode = phase._get_subsystem(self._rhs_source)
        ode_inputs = get_promoted_vars(ode, iotypes='input')

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, targets, dynamic in [('t', options['targets'], True),
                                       ('t_phase', options['time_phase_targets'], True),
                                       ('dt_dstau', options['dt_dstau_targets'], True)]:
            if targets:
                src_idxs = self.grid_data.subset_node_indices['all'] if dynamic else None
                phase.connect(name, [f'rhs_all.{t}' for t in targets], src_indices=src_idxs,
                              flat_src_indices=True if dynamic else None)

        for name, targets in [('t_initial', options['t_initial_targets']),
                              ('t_duration', options['t_duration_targets'])]:
            for t in targets:
                shape = ode_inputs[t]['shape']

                if shape == (1,):
                    src_idxs = None
                    flat_src_idxs = None
                    src_shape = None
                else:
                    src_idxs = np.zeros(self.grid_data.subset_num_nodes['all'])
                    flat_src_idxs = True
                    src_shape = (1,)

                phase.promotes('rhs_all', inputs=[(t, name)], src_indices=src_idxs,
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
        super(Radau, self).configure_controls(phase)

        for name, options in phase.control_options.items():
            if options['targets']:
                phase.connect(f'control_values:{name}', [f'rhs_all.{t}' for t in options['targets']])

            if options['rate_targets']:
                phase.connect(f'control_rates:{name}_rate',
                              [f'rhs_all.{t}' for t in options['rate_targets']])

            if options['rate2_targets']:
                phase.connect(f'control_rates:{name}_rate2',
                              [f'rhs_all.{t}' for t in options['rate2_targets']])

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).setup_ode(phase)

        grid_data = self.grid_data

        ode_sys = _make_ode_system(ode_class=phase.options['ode_class'],
                                   num_nodes=grid_data.subset_num_nodes['all'],
                                   ode_init_kwargs=phase.options['ode_init_kwargs'],
                                   calc_exprs=phase._calc_exprs,
                                   parameter_options=phase.parameter_options)

        phase.add_subsystem('rhs_all',
                            subsys=ode_sys)

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).configure_ode(phase)

        grid_data = self.grid_data
        map_input_indices_to_disc = grid_data.input_maps['state_input_to_disc']
        ode_inputs = get_promoted_vars(phase.rhs_all, 'input')

        for name, options in phase.state_options.items():

            targets = get_targets(ode_inputs, name=name, user_targets=options['targets'])
            if targets:
                phase.connect(f'states:{name}',
                              [f'rhs_all.{tgt}' for tgt in targets],
                              src_indices=om.slicer[map_input_indices_to_disc, ...])

    def setup_defects(self, phase):
        """
        Create the continuity_comp to house the defects.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).setup_defects(phase)

        if any(self._requires_continuity_constraints(phase)):
            phase.add_subsystem('continuity_comp',
                                RadauPSContinuityComp(grid_data=self.grid_data,
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
        super(Radau, self).configure_defects(phase)

        for name in phase.state_options:
            phase.connect(f'state_interp.staterate_col:{name}',
                          f'collocation_constraint.f_approx:{name}')

            rate_src_path, src_idxs = self._get_rate_source_path(name, 'col', phase)
            phase.connect(rate_src_path,
                          f'collocation_constraint.f_computed:{name}',
                          src_indices=src_idxs)

        any_state_cnty, any_control_cnty, any_control_rate_cnty = self._requires_continuity_constraints(phase)

        if any((any_state_cnty, any_control_cnty, any_control_rate_cnty)):
            phase._get_subsystem('continuity_comp').configure_io()

    def _get_rate_source_path(self, state_name, nodes, phase):
        """
        Return the rate source location and indices for a given state name.

        Parameters
        ----------
        state_name : str
            Name of the state.
        nodes : str
            One of ['col', 'all'].
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
            raise ValueError(f"state '{state_name}' in phase '{phase.name}' was not given a "
                             "rate_source")

        # Note the rate source must be shape-compatible with the state
        var_type = phase.classify_var(var)

        # Determine the path to the variable
        if var_type == 't':
            rate_path = 't'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 't_phase':
            rate_path = 't_phase'
            node_idxs = gd.subset_node_indices[nodes]
        elif var_type == 'state':
            rate_path = f'states:{var}'
            # Find the state_input indices which occur at segment endpoints, and repeat them twice
            state_input_idxs = gd.subset_node_indices['state_input']
            repeat_idxs = np.ones_like(state_input_idxs)
            if self.options['compressed']:
                segment_end_idxs = gd.subset_node_indices['segment_ends'][1:-1]
                # Repeat nodes that are on segment bounds (but not the first or last nodes in the phase)
                nodes_to_repeat = list(set(state_input_idxs).intersection(segment_end_idxs))
                # Now find these nodes in the state input indices
                idxs_of_ntr_in_state_inputs = np.where(np.isin(state_input_idxs, nodes_to_repeat))[0]
                # All state input nodes are used once, but nodes_to_repeat are used twice
                repeat_idxs[idxs_of_ntr_in_state_inputs] = 2
            # Now we have a way of mapping the state input indices to all nodes
            map_input_node_idxs_to_all = np.repeat(np.arange(gd.subset_num_nodes['state_input'],
                                                             dtype=int), repeats=repeat_idxs)
            # Now select the subset of nodes we want to use.
            node_idxs = map_input_node_idxs_to_all[gd.subset_node_indices[nodes]]
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
        else:
            # Failed to find variable, assume it is in the ODE
            rate_path = f'rhs_all.{var}'
            node_idxs = gd.subset_node_indices[nodes]

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
            path = f'states:{var}'
            src_units = phase.state_options[var]['units']
            src_shape = phase.state_options[var]['shape']

            # Find the state_input indices which occur at segment endpoints, and repeat them twice
            state_input_idxs = gd.subset_node_indices['state_input']
            repeat_idxs = np.ones_like(state_input_idxs)
            if self.options['compressed']:
                segment_end_idxs = gd.subset_node_indices['segment_ends'][1:-1]
                # Repeat nodes that are on segment bounds (but not the first or last nodes in the phase)
                nodes_to_repeat = list(set(state_input_idxs).intersection(set(segment_end_idxs)))
                # Now find these nodes in the state input indices
                idxs_of_ntr_in_state_inputs = np.where(np.isin(state_input_idxs, nodes_to_repeat))[0]
                # All state input nodes are used once, but nodes_to_repeat are used twice
                repeat_idxs[idxs_of_ntr_in_state_inputs] = 2
            # Now we have a way of mapping the state input indices to all nodes
            map_input_node_idxs_to_all = np.repeat(np.arange(gd.subset_num_nodes['state_input'],
                                                             dtype=int), repeats=repeat_idxs)
            # Now select the subset of nodes we want to use.
            node_idxs = map_input_node_idxs_to_all[gd.subset_node_indices['all']]
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
            # Timeseries are never a static_target
            node_idxs = np.zeros(gd.subset_num_nodes['all'], dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            path = f'rhs_all.{var}'
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(f'ODE output {var} is tagged with "dymos.static_output" and '
                                   'cannot be a timeseries output.')

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
            Parameter name.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        connection_info = []

        if name in phase.parameter_options:
            options = phase.parameter_options[name]
            for tgt in options['targets']:
                if tgt in options['static_targets']:
                    src_idxs = np.squeeze(get_src_indices_by_row([0], options['shape']), axis=0)
                else:
                    src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                    src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                    if options['shape'] == (1,):
                        src_idxs = src_idxs.ravel()

                connection_info.append((f'rhs_all.{tgt}', (src_idxs,)))

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
        any_control_continuity = any_control_continuity and num_seg > 1
        any_rate_continuity = any([opts['rate_continuity'] or opts['rate2_continuity']
                                   for opts in phase.control_options.values()])
        any_rate_continuity = any_rate_continuity and num_seg > 1

        return any_state_continuity, any_control_continuity, any_rate_continuity
