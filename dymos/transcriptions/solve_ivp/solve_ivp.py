import numpy as np

import openmdao.api as om

from ..transcription_base import TranscriptionBase
from .components import SegmentSimulationComp, SegmentStateMuxComp, \
    SolveIVPControlGroup, SolveIVPTimeseriesOutputComp
from ..common import TimeComp, TimeseriesOutputGroup
from ...utils.misc import get_rate_units
from ...utils.introspection import get_promoted_vars, get_targets, get_source_metadata
from ...utils.indexing import get_src_indices_by_row


class SolveIVP(TranscriptionBase):
    """
    The SolveIVP Transcription class.

    SolveIVP transcription in Dymos uses the scipy.simulate.solve_ivp method to explicitly integrate
    the states from the phase initial time to the phase final time.

    SolveIVP transcription does not currently support optimization since it does not propagate
    analytic derivatives through the ODE.

    Parameters
    ----------
    grid_data : GridData
        Grid data for this phases.
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, grid_data=None, **kwargs):
        om.issue_warning('The SolveIVP transcription is deprecated. The simulate methods in Dymos now uses '
                         'the ExplicitShooting transcription without derivative propagation to achieve the same'
                         'functionality. SolveIVP will be removed in a future version of Dymos.',
                         category=om.OMDeprecationWarning)
        super(SolveIVP, self).__init__(**kwargs)
        self.grid_data = grid_data
        self._rhs_source = 'ode'

    def initialize(self):
        """
        Declare transcription options.
        """
        super(SolveIVP, self).initialize()

        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

        self.options.declare('reports', default=False, desc='Reports setting for the subproblem.')

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        pass

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_options = phase.time_options
        time_units = time_options['units']
        num_seg = self.grid_data.num_segments
        grid_data = self.grid_data
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        super(SolveIVP, self).setup_time(phase)

        if output_nodes_per_seg is None:
            # Case 1:  Compute times at 'all' node set.
            num_nodes = grid_data.num_nodes
            node_ptau = grid_data.node_ptau
            node_dptau_dstau = grid_data.node_dptau_dstau
        else:
            # Case 2:  Compute times at n equally distributed points per segment.
            num_nodes = num_seg * output_nodes_per_seg
            node_stau = np.linspace(-1, 1, output_nodes_per_seg)
            node_ptau = np.empty(0, )
            node_dptau_dstau = np.empty(0, )
            # Append our nodes in phase tau space
            for iseg in range(num_seg):
                v0 = grid_data.segment_ends[iseg]
                v1 = grid_data.segment_ends[iseg + 1]
                node_ptau = np.concatenate((node_ptau, v0 + 0.5 * (node_stau + 1) * (v1 - v0)))
                node_dptau_dstau = np.concatenate((node_dptau_dstau,
                                                   0.5 * (v1 - v0) * np.ones_like(node_stau)))

        time_comp = TimeComp(num_nodes=num_nodes, node_ptau=node_ptau,
                             node_dptau_dstau=node_dptau_dstau, units=time_units)

        phase.add_subsystem('time', time_comp, promotes=['*'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(SolveIVP, self).configure_time(phase)
        num_seg = self.grid_data.num_segments
        grid_data = self.grid_data
        output_nodes_per_seg = self.options['output_nodes_per_seg']
        ode = self._get_ode(phase)
        time_name = phase.time_options['name']

        phase.time.configure_io()

        for i in range(num_seg):
            if output_nodes_per_seg is None:
                i1, i2 = grid_data.subset_segment_indices['all'][i, :]
                src_idxs = grid_data.subset_node_indices['all'][i1:i2]
            else:
                src_idxs = np.arange(i * output_nodes_per_seg, output_nodes_per_seg * (i + 1),
                                     dtype=int)
            phase.connect('t', f'segment_{i}.{time_name}', src_indices=src_idxs, flat_src_indices=True)
            phase.connect('t_phase', f'segment_{i}.t_phase', src_indices=src_idxs,
                          flat_src_indices=True)

            phase.segments.promotes(f'segment_{i}', inputs=['t_initial', 't_duration'])

        options = phase.time_options

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, targets, dynamic in [('t', options['targets'], True),
                                       ('t_phase', options['time_phase_targets'], True)]:
            if targets:
                phase.connect(name, [f'ode.{t}' for t in targets])

        for name, targets, dynamic in [('t_initial', options['t_initial_targets'], False),
                                       ('t_duration', options['t_duration_targets'], False)]:

            shape, units, static_target = get_target_metadata(ode, name=name,
                                                              user_targets=targets,
                                                              user_units=options['units'],
                                                              user_shape=(1,))
            if shape == (1,):
                src_idxs = None
                flat_src_idxs = None
                src_shape = None
            else:
                src_idxs = np.zeros(self.grid_data.subset_num_nodes['all'])
                flat_src_idxs = True
                src_shape = (1,)

            for t in targets:
                phase.promotes('ode', inputs=[(t, name)], src_indices=src_idxs,
                               flat_src_indices=flat_src_idxs, src_shape=src_shape)
            if targets:
                phase.set_input_defaults(name=name,
                                         val=np.ones((1,)),
                                         units=options['units'])

        phase.connect('dt_dstau', 'timeseries.dt_dstau')

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase.add_subsystem('indep_states', om.IndepVarComp(),
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
        num_seg = self.grid_data.num_segments

        for state_name, options in phase.state_options.items():
            phase.indep_states.add_output(f'initial_states:{state_name}',
                                          val=np.ones(((1,) + options['shape'])),
                                          units=options['units'])

        for state_name, options in phase.state_options.items():
            # Connect the initial state to the first segment
            src_idxs = get_src_indices_by_row([0], options['shape'])

            phase.connect(f'initial_states:{state_name}',
                          f'segment_0.initial_states:{state_name}',
                          src_indices=(src_idxs,), flat_src_indices=True)

            phase.connect(f'segment_0.states:{state_name}',
                          f'state_mux_comp.segment_0_states:{state_name}')

            if options['targets']:
                phase.connect(f'state_mux_comp.states:{state_name}',
                              [f'ode.{t}' for t in options['targets']])

            # Connect the final state in segment n to the initial state in segment n + 1
            for i in range(1, num_seg):
                if self.options['output_nodes_per_seg'] is None:
                    nnps_i = self.grid_data.subset_num_nodes_per_segment['all'][i]
                else:
                    nnps_i = self.options['output_nodes_per_seg']

                src_idxs = get_src_indices_by_row([nnps_i-1], shape=options['shape'])
                phase.connect(f'segment_{i - 1}.states:{state_name}',
                              f'segment_{i}.initial_states:{state_name}',
                              src_indices=(src_idxs,), flat_src_indices=True)

                phase.connect(f'segment_{i}.states:{state_name}',
                              f'state_mux_comp.segment_{i}_states:{state_name}')

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data
        num_seg = gd.num_segments

        segments_group = phase.add_subsystem(name='segments', subsys=om.Group(),
                                             promotes_outputs=['*'], promotes_inputs=['*'])

        for i in range(num_seg):
            seg_i_comp = SegmentSimulationComp(
                index=i,
                simulate_options=phase.simulate_options,
                grid_data=self.grid_data,
                ode_class=phase.options['ode_class'],
                ode_init_kwargs=phase.options['ode_init_kwargs'],
                time_options=phase.time_options,
                state_options=phase.state_options,
                control_options=phase.control_options,
                parameter_options=phase.parameter_options,
                output_nodes_per_seg=self.options['output_nodes_per_seg'],
                reports=self.options['reports'])

            segments_group.add_subsystem(f'segment_{i}', subsys=seg_i_comp)

        # scipy.integrate.solve_ivp does not actually evaluate the ODE at the desired output points,
        # but just returns the time and interpolated integrated state values there instead. We need
        # to instantiate a second ODE group that will call the ODE at those points so that we can
        # accurately obtain timeseries for ODE outputs.
        phase.add_subsystem('state_mux_comp',
                            SegmentStateMuxComp(grid_data=gd, state_options=phase.state_options,
                                                output_nodes_per_seg=self.options['output_nodes_per_seg']))

        if self.options['output_nodes_per_seg'] is None:
            self.num_output_nodes = gd.subset_num_nodes['all']
        else:
            self.num_output_nodes = num_seg * self.options['output_nodes_per_seg']

        phase.add_subsystem('ode', phase.options['ode_class'](num_nodes=self.num_output_nodes,
                                                              **phase.options['ode_init_kwargs']))

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data
        num_seg = gd.num_segments

        for i in range(num_seg):
            seg_comp = phase.segments._get_subsystem(f'segment_{i}')
            seg_comp.configure_io()

    def setup_controls(self, phase):
        """
        Setup the control group.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        phase._check_control_options()

        if phase.control_options:
            control_group = SolveIVPControlGroup(control_options=phase.control_options,
                                                 time_units=phase.time_options['units'],
                                                 grid_data=self.grid_data,
                                                 output_nodes_per_seg=output_nodes_per_seg)

            phase.add_subsystem('control_group',
                                subsys=control_group,
                                promotes=['*controls:*', '*control_values:*', '*control_values_all:*',
                                          '*control_rates:*'])

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        if phase.control_options:
            phase.control_group.configure_io()
            phase.connect('dt_dstau', 'control_group.dt_dstau')

        for name, options in phase.control_options.items():
            for i in range(grid_data.num_segments):
                i1, i2 = grid_data.subset_segment_indices['control_disc'][i, :]
                seg_idxs = grid_data.subset_node_indices['control_disc'][i1:i2]
                src_idxs = get_src_indices_by_row(row_idxs=seg_idxs, shape=options['shape'])
                phase.connect(src_name=f'control_values_all:{name}',
                              tgt_name=f'segment_{i}.controls:{name}',
                              src_indices=(src_idxs,), flat_src_indices=True)

            if options['targets']:
                phase.connect(f'control_values:{name}', [f'ode.{t}' for t in options['targets']])

            if options['rate_targets']:
                phase.connect(f'control_rates:{name}_rate', [f'ode.{t}' for t in options['rate_targets']])

            if options['rate2_targets']:
                phase.connect(f'control_rates:{name}_rate2', [f'ode.{t}' for t in options['rate2_targets']])

    def configure_parameters(self, phase):
        """
        Configure parameter promotion.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(SolveIVP, self).configure_parameters(phase)

        gd = self.grid_data

        # We also need to take care of the segments.
        segs = phase._get_subsystem('segments')

        for name, options in phase.parameter_options.items():
            # prom_name = f'parameters:{name}'
            shape = options['shape']
            units = options['units']

            for i in range(gd.num_segments):
                seg_comp = segs._get_subsystem(f'segment_{i}')
                seg_comp.add_input(name=f'parameters:{name}', val=np.ones(shape), units=units,
                                   desc=f'values of parameter {name}.')
                # phase.connect(f'parameter_vals:{name}', f'segment_{i}.{prom_name}')

    def setup_defects(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_defects(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_objective(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_path_constraints(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_boundary_constraints(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_solvers(self, phase):
        """
        Not used in SolveIVP.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_solvers(self, phase):
        """
        Not used in SolveIVP.

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
        # Check if timeseries contains an expression that needs to be evaluated
        for _, output_options in phase._timeseries['timeseries']['outputs'].items():
            if output_options['is_expr']:
                has_expr = True
                break
            else:
                has_expr = False

        timeseries_comp = \
            SolveIVPTimeseriesOutputComp(input_grid_data=gd,
                                         output_nodes_per_seg=self.options['output_nodes_per_seg'],
                                         time_units=phase.time_options['units'])

        timeseries_group = TimeseriesOutputGroup(has_expr=has_expr, timeseries_output_comp=timeseries_comp)
        phase.add_subsystem('timeseries', subsys=timeseries_group)

        # Remove all subsequent timeseries
        for ts_name in list(phase._timeseries.keys()):
            if ts_name != 'timeseries':
                phase._timeseries.pop(ts_name)

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            The name of the parameter for which connection information is desired.
        phase : dymos.Phase
            The phase object to which this transcription applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        connection_info = []
        num_seg = self.grid_data.num_segments
        output_nodes_per_seg = self.options['output_nodes_per_seg']
        num_final_ode_nodes = self.grid_data.subset_num_nodes['all'] \
            if output_nodes_per_seg is None else num_seg * output_nodes_per_seg

        if name in phase.parameter_options:
            options = phase.parameter_options[name]

            static = options['static_target']
            shape = options['shape']

            # Get connections to each segment
            gd = self.grid_data

            for i in range(gd.num_segments):
                phase.connect(f'parameter_vals:{name}', f'segment_{i}.parameters:{name}')

            # Connections to the final ODE
            ode_tgts = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])
            if not static:
                src_idxs_raw = np.zeros(num_final_ode_nodes, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                if shape == (1,):
                    src_idxs = src_idxs.ravel()
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                src_idxs = np.squeeze(src_idxs, axis=0)

            connection_info.append(([f'ode.{tgt}' for tgt in ode_tgts], (src_idxs,)))

        return connection_info

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
        var_type = phase.classify_var(var)
        time_units = phase.time_options['units']

        transcription = phase.options['transcription']
        ode = transcription._get_ode(phase)
        ode_outputs = get_promoted_vars(ode, 'output')

        # The default for node_idxs, applies to everything except states and parameters.
        node_idxs = None
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
            path = f'state_mux_comp.states:{var}'
            src_units = phase.state_options[var]['units']
            src_shape = phase.state_options[var]['shape']
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
        elif var_type == 'parameter':
            path = f'parameter_vals:{var}'
            num_seg = self.grid_data.num_segments
            node_idxs = np.zeros(num_seg * self.options['output_nodes_per_seg'], dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            path = f'ode.{var}'
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(f'ODE output {var} is tagged with "dymos.static_output" and cannot be a '
                                   f'timeseries output.')

        src_idxs = None if node_idxs is None else om.slicer[node_idxs, ...]

        meta['src'] = path
        meta['src_idxs'] = src_idxs
        meta['units'] = src_units
        meta['shape'] = src_shape

        return meta

    def _get_num_timeseries_nodes(self):
        """
        Returns the number of nodes in the default timeseries for this transcription.

        Returns
        -------
        int
            The number of nodes in the default timeseries for this transcription.
        """
        if self.options['output_nodes_per_seg'] is None:
            output_nodes = self.grid_data.num_segments
        else:
            output_nodes = self.grid_data.num_segments * self.options['output_nodes_per_seg']
        return output_nodes
