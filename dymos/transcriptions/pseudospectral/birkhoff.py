import numpy as np

import openmdao.api as om

from ..transcription_base import TranscriptionBase
from ..common import TimeComp, TimeseriesOutputGroup
from .components import StateIndependentsComp, PseudospectralTimeseriesOutputComp, BirkhoffCollocationComp

from ..grid_data import BirkhoffGrid
from dymos.utils.misc import get_rate_units, reshape_val
from dymos.utils.introspection import get_promoted_vars, get_source_metadata, get_targets
from dymos.utils.indexing import get_src_indices_by_row


class Birkhoff(TranscriptionBase):
    def __init__(self, **kwargs):
        super(Birkhoff, self).__init__(**kwargs)
        self._rhs_source = 'rhs_all'

    def initialize(self):
        """
        Declare transcription options.
        """
        self.options.declare('grid', types=(BirkhoffGrid, str),
                             allow_none=True, default=None,
                             desc='The grid distribution used to layout the control inputs and provide the default '
                                  'output nodes.')

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        if self.options['grid'] in ('gauss-lobatto', None):
            self.grid_data = BirkhoffGrid(num_segments=self.options['num_segments'],
                                          nodes_per_seg=self.options['order'],
                                          segment_ends=self.options['segment_ends'],
                                          compressed=self.options['compressed'])
        else:
            self.grid_data = self.options['grid']

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        super(Birkhoff, self).setup_time(phase)

        time_comp = TimeComp(num_nodes=grid_data.num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau,
                             units=phase.time_options['units'],
                             initial_val=phase.time_options['initial_val'],
                             duration_val=phase.time_options['duration_val'])

        phase.add_subsystem('time', time_comp, promotes_inputs=['*'], promotes_outputs=['*'])

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
        super(Birkhoff, self).configure_time(phase)
        phase.time.configure_io()
        options = phase.time_options
        ode = phase._get_subsystem(self._rhs_source)
        ode_inputs = get_promoted_vars(ode, iotypes='input')

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, targets, dynamic in [('t', options['targets'], True),
                                       ('t_phase', options['time_phase_targets'], True)]:
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

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        self.any_solved_segs = False
        self.any_connected_opt_segs = False
        for options in phase.state_options.values():
            if options['input_initial']:
                self.any_connected_opt_segs = True

        if self.any_connected_opt_segs:
            indep = StateIndependentsComp(grid_data=grid_data, state_options=phase.state_options)
        else:
            indep = om.IndepVarComp()

        num_connected = len([s for (s, opts) in phase.state_options.items() if opts['input_initial']])
        prom_inputs = ['initial_states:*'] if num_connected > 0 else None
        phase.add_subsystem('indep_states', indep, promotes_inputs=prom_inputs,
                            promotes_outputs=['*'])

        indep_state_rates = om.IndepVarComp()

        phase.add_subsystem('indep_state_rates', indep_state_rates,
                            promotes_outputs=['*'])

    def configure_states(self, phase):
        """
        Configure state connections post-introspection.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """

        super().configure_states(phase)
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']
        indep = phase.indep_states
        indep_state_rates = phase.indep_state_rates

        # state_idx_map holds the node indices provided by the solver (solver) and those
        # that are independent variables (indep)
        self.state_idx_map = {}

        time_units = phase.time_options['units']

        # add all the des-vars (either from the IndepVarComp or from the indep-var-like
        # outputs of the collocation comp)
        for name, options in phase.state_options.items():
            shape = options['shape']
            units = options['units']
            # In certain cases, we put an output on the IVC.
            if isinstance(indep, om.IndepVarComp):
                default_val = reshape_val(options['val'], shape, num_state_input_nodes)
                indep.add_output(name=f'states:{name}',
                                 shape=(num_state_input_nodes,) + shape,
                                 val=default_val,
                                 units=units)

                indep_state_rates.add_output(name=f'state_rates:{name}',
                                             shape=(num_state_input_nodes,) + shape,
                                             val=default_val,
                                             units=units)

                phase.add_design_var(name=f'state_rates:{name}')

            if options['opt']:
                # Add the states as design variables.
                #
                # In the case of optimizer-driven collocation, this includes the values at all
                # nodes (excluding initial and/or final if fix_initial and/or fix_final is specified).
                #
                desvar_node_idxs = np.asarray(np.arange(num_state_input_nodes), dtype=int)

                # This matrix will contain 1's for every index that is to be a design variable,
                # otherwise the value will be zero
                state_input_shape = (num_state_input_nodes,) + options['shape']
                idx_mask = np.zeros(state_input_shape, dtype=int)
                idx_mask[desvar_node_idxs, ...] = 1

                if not (options['fix_initial'] or options['input_initial']):
                    ilb = options['lower']
                    iub = options['upper']

                    if options['initial_bounds'] is not None:
                        ilb = options['initial_bounds'][0]
                        iub = options['initial_bounds'][1]

                    phase.add_design_var(name=f'initial_states:{name}',
                                         lower=ilb,
                                         upper=iub,
                                         scaler=options['scaler'],
                                         adder=options['adder'],
                                         ref0=options['ref0'],
                                         ref=options['ref'])

                if not (options['fix_final'] or options['input_final']):
                    flb = options['lower']
                    fub = options['upper']

                    if options['final_bounds'] is not None:
                        flb = options['final_bounds'][0]
                        fub = options['final_bounds'][1]

                    phase.add_design_var(name=f'final_states:{name}',
                                         lower=flb,
                                         upper=fub,
                                         scaler=options['scaler'],
                                         adder=options['adder'],
                                         ref0=options['ref0'],
                                         ref=options['ref'])

                phase.add_design_var(name=f'states:{name}',
                                     lower=options['lower'],
                                     upper=options['upper'],
                                     scaler=options['scaler'],
                                     adder=options['adder'],
                                     ref0=options['ref0'],
                                     ref=options['ref'])

        if isinstance(indep, StateIndependentsComp):
            indep.configure_io(self.state_idx_map)

        if self.any_connected_opt_segs:
            for name, options in phase.state_options.items():
                if options['solve_segments']:
                    phase.connect(f'collocation_constraint.defects:{name}',
                                  f'indep_states.defects:{name}')

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_controls(phase)

        if phase.control_options:
            phase.control_group.configure_io()
            phase.promotes('control_group',
                           any=['controls:*', 'control_values:*', 'control_rates:*'])

            phase.connect('dt_dstau', 'control_group.dt_dstau')

        for name, options in phase.control_options.items():
            if options['targets']:
                phase.connect(f'control_values:{name}', [f'rhs_all.{t}' for t in options['targets']])

            if options['rate_targets']:
                phase.connect(f'control_rates:{name}_rate',
                              [f'rhs_all.{t}' for t in options['rate_targets']])

            if options['rate2_targets']:
                phase.connect(f'control_rates:{name}_rate2',
                              [f'rhs_all.{t}' for t in options['rate2_targets']])

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(Birkhoff, self).configure_polynomial_controls(phase)

        ode_inputs = get_promoted_vars(self._get_ode(phase), 'input')

        for name, options in phase.polynomial_control_options.items():
            targets = get_targets(ode=ode_inputs, name=name, user_targets=options['targets'])
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
        phase.add_subsystem('collocation_constraint',
                            BirkhoffCollocationComp(grid_data=self.grid_data,
                                                    state_options=phase.state_options,
                                                    time_units=phase.time_options['units']),
                            promotes_inputs=['states:*', 'state_rates:*', 'initial_states:*',
                                             'final_states:*'])

    def configure_defects(self, phase):
        """
        Configure the continuity_comp and connect the collocation constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        phase.connect('dt_dstau', ('collocation_constraint.dt_dstau'),
                      src_indices=grid_data.subset_node_indices['col'], flat_src_indices=True)

        phase.collocation_constraint.configure_io()

        for name in phase.state_options:
            rate_src_path, src_idxs = self._get_rate_source_path(name, 'col', phase)

            # phase.connect(f'states:{name}',
            #               f'collocation_constraint.states:{name}')
            #
            # phase.connect(f'state_rates:{name}',
            #               f'collocation_constraint.state_rates:{name}')

            phase.connect(rate_src_path,
                          f'collocation_constraint.f_computed:{name}')

    def setup_duration_balance(self, phase):
        """
        Setup the implicit computation of the phase duration.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_duration_balance(self, phase):
        """
        Configure the implicit computation of the phase duration.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_solvers(self, phase):
        """
        Setup the solvers.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_solvers(self, phase):
        """
        Configure the solvers.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data

        for name, options in phase._timeseries.items():
            has_expr = False
            for _, output_options in options['outputs'].items():
                if output_options['is_expr']:
                    has_expr = True
                    break

            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = PseudospectralTimeseriesOutputComp(input_grid_data=gd,
                                                                 output_grid_data=ogd,
                                                                 output_subset=options['subset'],
                                                                 time_units=phase.time_options['units'])
            timeseries_group = TimeseriesOutputGroup(has_expr=has_expr, timeseries_output_comp=timeseries_comp)
            phase.add_subsystem(name, subsys=timeseries_group)

            phase.connect('dt_dstau', f'{name}.dt_dstau', flat_src_indices=True)

    def _get_objective_src(self, var, loc, phase, ode_outputs=None):
        """
        Return the path to the variable that will be used as the objective.

        Parameters
        ----------
        var : str
            Name of the variable to be used as the objective.
        loc : str
            The location of the objective in the phase ['initial', 'final'].
        phase : dymos.Phase
            Phase object containing in which the objective resides.
        ode_outputs : dict or None
            A dictionary of ODE outputs as returned by get_promoted_vars.

        Returns
        -------
        obj_path : str
            Path to the source.
        shape : tuple
            Source shape.
        units : str
            Source units.
        linear : bool
            True if the objective quantity1 is linear.
        """
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

        if ode_outputs is None:
            ode_outputs = get_promoted_vars(phase._get_subsystem(self._rhs_source), 'output')

        if var_type == 't':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 't'
        elif var_type == 't_phase':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 't_phase'
        elif var_type == 'state':
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            solve_segments = phase.state_options[var]['solve_segments']
            connected_initial = phase.state_options[var]['input_initial']
            if not solve_segments and not connected_initial:
                linear = True
            elif solve_segments in {'forward'} and not connected_initial and loc == 'initial':
                linear = True
            elif solve_segments == 'backward' and loc == 'final':
                linear = True
            else:
                linear = False
            constraint_path = f'states:{var}'
        elif var_type == 'indep_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = True
            constraint_path = f'control_values:{var}'
        elif var_type == 'input_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            constraint_path = f'control_values:{var}'
        elif var_type == 'indep_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = True
            constraint_path = f'polynomial_control_values:{var}'
        elif var_type == 'input_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = False
            constraint_path = f'polynomial_control_values:{var}'
        elif var_type == 'parameter':
            shape = phase.parameter_options[var]['shape']
            units = phase.parameter_options[var]['units']
            linear = True
            constraint_path = f'parameter_vals:{var}'
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5] if var_type == 'control_rate' else var[:-6]
            shape = phase.control_options[control_var]['shape']
            control_units = phase.control_options[control_var]['units']
            d = 2 if var_type == 'control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            constraint_path = f'control_rates:{var}'
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5]
            shape = phase.polynomial_control_options[control_var]['shape']
            control_units = phase.polynomial_control_options[control_var]['units']
            d = 2 if var_type == 'polynomial_control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            constraint_path = f'polynomial_control_rates:{var}'
        elif var_type == 'timeseries_exec_comp_output':
            shape = (1,)
            units = None
            constraint_path = f'timeseries.timeseries_exec_comp.{var}'
            linear = False
        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
            constraint_path = f'{self._rhs_source}.{var}'
            meta = get_source_metadata(ode_outputs, var, user_units=None, user_shape=None)
            shape = meta['shape']
            units = meta['units']
            linear = False

        return constraint_path, shape, units, linear

    def _get_num_timeseries_nodes(self):
        """
        Returns the number of nodes in the default timeseries for this transcription.

        Returns
        -------
        int
            The number of nodes in the default timeseries for this transcription.
        """
        return self.grid_data.num_nodes

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
                idxs_of_ntr_in_state_inputs = np.where(np.in1d(state_input_idxs, nodes_to_repeat))[0]
                # All state input nodes are used once, but nodes_to_repeat are used twice
                repeat_idxs[idxs_of_ntr_in_state_inputs] = 2
            # Now we have a way of mapping the state input indices to all nodes
            map_input_node_idxs_to_all = np.repeat(np.arange(gd.subset_num_nodes['state_input'],
                                                             dtype=int), repeats=repeat_idxs)
            # Now select the subset of nodes we want to use.
            node_idxs = map_input_node_idxs_to_all[gd.subset_node_indices[nodes]]
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
            dynamic = not phase.parameter_options[var]['static_target']
            if dynamic:
                node_idxs = np.zeros(gd.subset_num_nodes[nodes], dtype=int)
            else:
                node_idxs = np.zeros(1, dtype=int)
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

            if gd.transcription == 'radau-ps':
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
                node_idxs = map_input_node_idxs_to_all[gd.subset_node_indices['all']]
        elif var_type in ['indep_control', 'input_control']:
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
        elif var_type in ['indep_polynomial_control', 'input_polynomial_control']:
            path = f'polynomial_control_values:{var}'
            src_units = phase.polynomial_control_options[var]['units']
            src_shape = phase.polynomial_control_options[var]['shape']
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            path = f'polynomial_control_rates:{control_name}_rate'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=1)
            src_shape = control['shape']
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            path = f'polynomial_control_rates:{control_name}_rate2'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=2)
            src_shape = control['shape']
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
                raise RuntimeError(
                    f'ODE output {var} is tagged with "dymos.static_output" and cannot be a timeseries output.')

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

