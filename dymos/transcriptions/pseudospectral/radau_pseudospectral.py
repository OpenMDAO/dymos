from collections import defaultdict
from fnmatch import filter
import warnings

import numpy as np
import openmdao.api as om
from openmdao.utils.general_utils import simple_warning

from .pseudospectral_base import PseudospectralBase
from ..common import RadauPSContinuityComp
from ...utils.misc import get_rate_units, get_source_metadata
from ...utils.introspection import get_targets
from ...utils.indexing import get_src_indices_by_row
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

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).configure_time(phase)
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
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
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
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
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
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).setup_ode(phase)

        ODEClass = phase.options['ode_class']
        grid_data = self.grid_data

        kwargs = phase.options['ode_init_kwargs']
        phase.add_subsystem('rhs_all',
                            subsys=ODEClass(num_nodes=grid_data.subset_num_nodes['all'],
                                            **kwargs))

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

        for name, options in phase.state_options.items():

            targets = get_targets(ode=phase.rhs_all, name=name, user_targets=options['targets'])
            if targets:
                phase.connect('states:{0}'.format(name),
                              ['rhs_all.{0}'.format(tgt) for tgt in targets],
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

        grid_data = self.grid_data
        if grid_data.num_segments > 1:
            phase.add_subsystem('continuity_comp',
                                RadauPSContinuityComp(grid_data=grid_data,
                                                      state_options=phase.state_options,
                                                      control_options=phase.control_options,
                                                      time_units=phase.time_options['units']),
                                promotes_inputs=['t_duration'])

    def configure_defects(self, phase):
        """
        Configure the continuity_comp and connect the collocation constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).configure_defects(phase)

        grid_data = self.grid_data
        if grid_data.num_segments > 1:
            phase.continuity_comp.configure_io()

        for name, options in phase.state_options.items():
            phase.connect('state_interp.staterate_col:{0}'.format(name),
                          'collocation_constraint.f_approx:{0}'.format(name))

            rate_src = options['rate_source']
            if rate_src in phase.parameter_options:
                # If the rate source is a parameter, which is an input, we need to promote
                # f_computed to the parameter name instead of connecting to it.
                shape = phase.parameter_options[rate_src]['shape']
                param_size = np.prod(shape)
                ncn = self.grid_data.subset_num_nodes['col']
                src_idxs = np.tile(np.arange(0, param_size, dtype=int), ncn)
                src_idxs = np.reshape(src_idxs, (ncn,) + shape)
                phase.promotes('collocation_constraint', inputs=[(f'f_computed:{name}', f'parameters:{rate_src}')],
                               src_indices=src_idxs, flat_src_indices=True, src_shape=shape)
            else:
                rate_src_path, src_idxs = self.get_rate_source_path(name, 'col', phase)
                phase.connect(rate_src_path,
                              'collocation_constraint.f_computed:{0}'.format(name),
                              src_indices=src_idxs)

    def configure_path_constraints(self, phase):
        """
        Handle the common operations for configuration of the path constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Radau, self).configure_path_constraints(phase)

        gd = self.grid_data

        for var, options in phase._path_constraints.items():
            constraint_kwargs = options.copy()
            con_name = constraint_kwargs.pop('constraint_name')
            src_idxs = None
            flat_src_idxs = False

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'time':
                src = 'time'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'time_phase':
                src = 'time_phase'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                src_idxs = get_src_indices_by_row(gd.input_maps['state_input_to_disc'], state_shape)
                flat_src_idxs = True
                src = f'states:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'indep_control':
                src = f'control_values:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'input_control':
                src = 'control_values:{0}'.format(var)
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'indep_polynomial_control':
                src = f'polynomial_control_values:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'input_polynomial_control':
                src = f'polynomial_control_values:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'control_rate':
                control_name = var[:-5]
                src = f'control_rates:{control_name}_rate'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                src = f'control_rates:{control_name}_rate2'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'polynomial_control_rate':
                control_name = var[:-5]
                src = f'polynomial_control_rates:{control_name}_rate'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type == 'polynomial_control_rate2':
                control_name = var[:-6]
                src = f'polynomial_control_rates:{control_name}_rate2'
                tgt = f'path_constraints.all_values:{con_name}'

            else:
                # Failed to find variable, assume it is in the ODE
                src = f'rhs_all.{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            phase.connect(src_name=src, tgt_name=tgt,
                          src_indices=src_idxs, flat_src_indices=flat_src_idxs)

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data
        time_units = phase.time_options['units']

        for timeseries_name in phase._timeseries:
            timeseries_comp = phase._get_subsystem(timeseries_name)

            timeseries_comp._add_output_configure('time',
                                                  shape=(1,),
                                                  units=time_units,
                                                  desc='',
                                                  src='time')

            timeseries_comp._add_output_configure('time_phase',
                                                  shape=(1,),
                                                  units=time_units,
                                                  desc='',
                                                  src='time_pahse')

            phase.connect(src_name='time', tgt_name=f'{timeseries_name}.input_values:time')

            phase.connect(src_name='time_phase', tgt_name=f'{timeseries_name}.input_values:time_phase')

            for state_name, options in phase.state_options.items():

                added_src = timeseries_comp._add_output_configure(f'states:{state_name}',
                                                                  shape=options['shape'],
                                                                  units=options['units'],
                                                                  desc=options['desc'],
                                                                  src=f'states:{state_name}')
                if added_src:
                    src_rows = gd.input_maps['state_input_to_disc']
                    phase.connect(src_name=f'states:{state_name}',
                                  tgt_name=f'{timeseries_name}.input_values:states:{state_name}',
                                  src_indices=om.slicer[src_rows, ...])

                rate_src = options['rate_source']
                if rate_src in phase.parameter_options:
                    # If the rate source is a parameter, which is an input, we need to promote
                    # the state rates input value to the parameter name instead of connecting to it.
                    nn = self.grid_data.subset_num_nodes['all']
                    shape = phase.parameter_options[rate_src]['shape']
                    param_size = np.prod(shape)
                    src_idxs = np.tile(np.arange(0, param_size, dtype=int), nn)
                    src_idxs = np.reshape(src_idxs, (nn,) + shape)
                    rate_src_path = f'parameters:{rate_src}'

                    added_src = timeseries_comp._add_output_configure(f'state_rates:{state_name}',
                                                                      shape=options['shape'],
                                                                      units=get_rate_units(
                                                                          options['units'],
                                                                          time_units),
                                                                      desc=f'rate of state {state_name}',
                                                                      src=rate_src_path)

                    if added_src:
                        phase.promotes(f'{timeseries_name}',
                                       inputs=[(f'input_values:state_rates:{state_name}',
                                                f'parameters:{rate_src}')],
                                       src_indices=src_idxs, flat_src_indices=True, src_shape=shape)

                else:
                    rate_src_path, src_idxs = self.get_rate_source_path(state_name, 'all', phase)

                    added_src = timeseries_comp._add_output_configure(f'state_rates:{state_name}',
                                                                      shape=options['shape'],
                                                                      units=get_rate_units(
                                                                          options['units'],
                                                                          time_units),
                                                                      desc=f'rate of state {state_name}',
                                                                      src=rate_src_path)
                    if added_src:
                        phase.connect(src_name=rate_src_path,
                                      tgt_name=f'{timeseries_name}.input_values:state_rates:{state_name}',
                                      src_indices=src_idxs)

            for control_name, options in phase.control_options.items():
                control_units = options['units']
                src_rows = gd.subset_node_indices['all']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])

                added_src = timeseries_comp._add_output_configure(f'controls:{control_name}',
                                                                  shape=options['shape'],
                                                                  units=control_units,
                                                                  desc=options['desc'],
                                                                  src=f'control_values:{control_name}')
                if added_src:
                    phase.connect(src_name=f'control_values:{control_name}',
                                  tgt_name=f'{timeseries_name}.input_values:controls:{control_name}',
                                  src_indices=src_idxs, flat_src_indices=True)

                # Control rates
                added_src = timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate',
                                                                  shape=options['shape'],
                                                                  units=get_rate_units(control_units,
                                                                                       time_units, deriv=1),
                                                                  desc=f'first time-derivative of {control_name}',
                                                                  src=f'control_rates:{control_name}_rate')
                if added_src:
                    phase.connect(src_name=f'control_rates:{control_name}_rate',
                                  tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate')

                # Control second derivatives
                added_src = timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate2',
                                                                  shape=options['shape'],
                                                                  units=get_rate_units(control_units,
                                                                                       time_units, deriv=2),
                                                                  desc=f'second time-derivative of {control_name}',
                                                                  src=f'control_rates:{control_name}_rate2')

                if added_src:
                    phase.connect(src_name=f'control_rates:{control_name}_rate2',
                                  tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate2')

            for control_name, options in phase.polynomial_control_options.items():
                src_rows = gd.subset_node_indices['all']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])
                control_units = options['units']

                # Control values
                added_src = timeseries_comp._add_output_configure(f'polynomial_controls:{control_name}',
                                                                  shape=options['shape'],
                                                                  units=control_units,
                                                                  desc=options['desc'],
                                                                  src=f'polynomial_control_values:{control_name}')

                if added_src:
                    phase.connect(src_name=f'polynomial_control_values:{control_name}',
                                  tgt_name=f'{timeseries_name}.input_values:polynomial_controls:{control_name}',
                                  src_indices=src_idxs, flat_src_indices=True)

                # Control rates
                added_src = timeseries_comp._add_output_configure(f'polynomial_control_rates:{control_name}_rate',
                                                                  shape=options['shape'],
                                                                  units=get_rate_units(control_units,
                                                                                       time_units, deriv=1),
                                                                  desc=f'first time-derivative of {control_name}',
                                                                  src=f'polynomial_control_rates:{control_name}_rate')
                if added_src:
                    phase.connect(src_name=f'polynomial_control_rates:{control_name}_rate',
                                  tgt_name=f'{timeseries_name}.input_values:polynomial_control_rates:{control_name}_rate')

                # Control second derivatives
                added_src = timeseries_comp._add_output_configure(f'polynomial_control_rates:{control_name}_rate2',
                                                                  shape=options['shape'],
                                                                  units=get_rate_units(control_units,
                                                                                       time_units, deriv=2),
                                                                  desc=f'second time-derivative of {control_name}',
                                                                  src=f'polynomial_control_rates:{control_name}_rate2')
                if added_src:
                    phase.connect(src_name=f'polynomial_control_rates:{control_name}_rate2',
                                  tgt_name=f'{timeseries_name}.input_values:polynomial_control_rates:{control_name}_rate2')

            for param_name, options in phase.parameter_options.items():
                if options['include_timeseries']:
                    prom_name = f'parameters:{param_name}'
                    tgt_name = f'input_values:parameters:{param_name}'

                    # Add output.
                    timeseries_comp = phase._get_subsystem(timeseries_name)
                    added_src = timeseries_comp._add_output_configure(prom_name,
                                                                      desc='',
                                                                      shape=options['shape'],
                                                                      units=options['units'],
                                                                      src=prom_name)
                    if added_src:
                        src_idxs_raw = np.zeros(gd.subset_num_nodes['all'], dtype=int)
                        src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                        phase.promotes(timeseries_name, inputs=[(tgt_name, prom_name)],
                                       src_indices=src_idxs, flat_src_indices=True)

            for var, options in phase._timeseries[timeseries_name]['outputs'].items():
                output_name = options['output_name']
                units = options.get('units', None)
                wildcard_units = options.get('wildcard_units', None)

                if '*' in var:  # match outputs from the ODE
                    ode_outputs = {opts['prom_name']: opts for (k, opts) in
                                   phase.rhs_all.get_io_metadata(iotypes=('output',)).items()}
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
                    if self.is_static_ode_output(v, phase, gd.subset_num_nodes['all']):
                        warnings.warn(f'Cannot add ODE output {v} to the timeseries output. It is '
                                      f'sized such that its first dimension != num_nodes.')
                        continue

                    try:
                        shape, units = get_source_metadata(phase.rhs_all, src=v,
                                                           user_units=units,
                                                           user_shape=options['shape'])
                    except ValueError:
                        raise ValueError(f'Timeseries output {v} is not a known variable in'
                                         f' the phase {phase.pathname} nor is it a known output of '
                                         f' the ODE.')

                    add_connection = timeseries_comp._add_output_configure(output_name, units,
                                                                           shape, desc='',
                                                                           src=f'rhs_all.{v}')

                    if add_connection:
                        phase.connect(src_name=f'rhs_all.{v}',
                                      tgt_name=f'{timeseries_name}.input_values:{output_name}')

    def get_rate_source_path(self, state_name, nodes, phase):
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
            raise ValueError('state \'{0}\' in phase \'{1}\' was not given a '
                             'rate_source'.format(state_name, phase.name))

        # Note the rate source must be shape-compatible with the state
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

        src_idxs = om.slicer[node_idxs, ...]

        return rate_path, src_idxs

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
            targets = get_targets(ode=phase.rhs_all, name=name, user_targets=options['targets'])

            static = options['static_target']
            shape = options['shape']

            if not static:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                if shape == (1,):
                    src_idxs = src_idxs.ravel()
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                src_idxs = np.squeeze(src_idxs, axis=0)

            rhs_all_tgts = [f'rhs_all.{t}' for t in targets]
            connection_info.append((rhs_all_tgts, src_idxs))

        return connection_info
