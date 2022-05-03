from fnmatch import filter
import warnings

import numpy as np

import openmdao.api as om

from ..transcription_base import TranscriptionBase
from .components import SegmentSimulationComp, SegmentStateMuxComp, \
    SolveIVPControlGroup, SolveIVPPolynomialControlGroup, SolveIVPTimeseriesOutputComp
from ..common import TimeComp
from ...utils.misc import get_rate_units
from ...utils.introspection import get_promoted_vars, get_targets, get_source_metadata, get_target_metadata
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
        ode_inputs = get_promoted_vars(ode, 'input')

        phase.time.configure_io()

        for i in range(num_seg):
            if output_nodes_per_seg is None:
                i1, i2 = grid_data.subset_segment_indices['all'][i, :]
                src_idxs = grid_data.subset_node_indices['all'][i1:i2]
            else:
                src_idxs = np.arange(i * output_nodes_per_seg, output_nodes_per_seg * (i + 1),
                                     dtype=int)
            phase.connect('time', f'segment_{i}.time', src_indices=src_idxs, flat_src_indices=True)
            phase.connect('time_phase', f'segment_{i}.time_phase', src_indices=src_idxs,
                          flat_src_indices=True)

            phase.segments.promotes(f'segment_{i}', inputs=['t_initial', 't_duration'])

        options = phase.time_options

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, targets, dynamic in [('time', options['targets'], True),
                                       ('time_phase', options['time_phase_targets'], True)]:
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
                polynomial_control_options=phase.polynomial_control_options,
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
                                promotes=['controls:*', 'control_values:*', 'control_values_all:*',
                                          'control_rates:*'])

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

    def setup_polynomial_controls(self, phase):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.polynomial_control_options:
            sys = SolveIVPPolynomialControlGroup(grid_data=self.grid_data,
                                                 polynomial_control_options=phase.polynomial_control_options,
                                                 time_units=phase.time_options['units'],
                                                 output_nodes_per_seg=self.options['output_nodes_per_seg'])
            phase.add_subsystem('polynomial_control_group', subsys=sys,
                                promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        # In transcription_base, we get the control units/shape from the target, and then call
        # configure on the control_group.
        super(SolveIVP, self).configure_polynomial_controls(phase)

        # Additional connections.
        for name, options in phase.polynomial_control_options.items():
            targets = options['targets']

            for iseg in range(self.grid_data.num_segments):
                phase.connect(src_name=f'polynomial_controls:{name}',
                              tgt_name=f'segment_{iseg}.polynomial_controls:{name}')

            if options['targets']:
                phase.connect(f'polynomial_control_values:{name}', [f'ode.{t}' for t in targets])

            if options['rate_targets']:
                phase.connect(f'polynomial_control_rates:{name}_rate',
                              [f'ode.{t}' for t in targets])

            if options['rate2_targets']:
                phase.connect(f'polynomial_control_rates:{name}_rate2',
                              [f'ode.{t}' for t in targets])

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

    def _configure_boundary_constraints(self, phase):
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

        timeseries_comp = \
            SolveIVPTimeseriesOutputComp(input_grid_data=gd,
                                         output_nodes_per_seg=self.options['output_nodes_per_seg'])

        phase.add_subsystem('timeseries', subsys=timeseries_comp)

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data
        num_seg = gd.num_segments
        output_nodes_per_seg = self.options['output_nodes_per_seg']
        time_units = phase.time_options['units']
        ode_outputs = get_promoted_vars(self._get_ode(phase), 'output')

        timeseries_name = 'timeseries'
        timeseries_comp = phase._get_subsystem(timeseries_name)

        timeseries_comp._add_output_configure('time', shape=(1,), units=time_units, desc='time')
        timeseries_comp._add_output_configure('time_phase', shape=(1,), units=time_units,
                                              desc='elapsed phase time')

        phase.connect(src_name='time', tgt_name='timeseries.all_values:time')
        phase.connect(src_name='time_phase', tgt_name='timeseries.all_values:time_phase')

        for name, options in phase.state_options.items():

            timeseries_comp._add_output_configure(f'states:{name}',
                                                  shape=options['shape'],
                                                  units=options['units'],
                                                  desc=options['desc'])

            timeseries_comp._add_output_configure(f'state_rates:{name}',
                                                  shape=options['shape'],
                                                  units=get_rate_units(options['units'],
                                                                       time_units, deriv=1),
                                                  desc=f'first time-derivative of state {name}')

            phase.connect(src_name=f'state_mux_comp.states:{name}',
                          tgt_name=f'timeseries.all_values:states:{name}')

            rate_src = phase.state_options[name]['rate_source']
            rate_path, node_idxs = self._get_rate_source_path(name, None, phase=phase)
            src_idxs = None if rate_src not in phase.parameter_options else om.slicer[node_idxs, ...]

            phase.connect(src_name=rate_path,
                          tgt_name=f'timeseries.all_values:state_rates:{name}',
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in phase.control_options.items():
            control_units = options['units']

            timeseries_comp._add_output_configure(f'controls:{name}',
                                                  shape=options['shape'],
                                                  units=control_units,
                                                  desc=options['desc'])

            phase.connect(src_name=f'control_values:{name}',
                          tgt_name=f'timeseries.all_values:controls:{name}')

            # Control rates
            timeseries_comp._add_output_configure(f'control_rates:{name}_rate',
                                                  shape=options['shape'],
                                                  units=get_rate_units(control_units, time_units,
                                                                       deriv=1),
                                                  desc=f'first time-derivative of control {name}')

            phase.connect(src_name=f'control_rates:{name}_rate',
                          tgt_name=f'timeseries.all_values:control_rates:{name}_rate')

            # Control second derivatives
            timeseries_comp._add_output_configure(f'control_rates:{name}_rate2',
                                                  shape=options['shape'],
                                                  units=get_rate_units(control_units, time_units,
                                                                       deriv=2),
                                                  desc=f'first time-derivative of control {name}')

            phase.connect(src_name=f'control_rates:{name}_rate2',
                          tgt_name=f'timeseries.all_values:control_rates:{name}_rate2')

        for name, options in phase.polynomial_control_options.items():
            control_units = options['units']
            timeseries_comp._add_output_configure(f'polynomial_controls:{name}',
                                                  shape=options['shape'],
                                                  units=control_units,
                                                  desc=options['desc'])

            phase.connect(src_name=f'polynomial_control_values:{name}',
                          tgt_name=f'timeseries.all_values:polynomial_controls:{name}')

            # Polynomial control rates
            timeseries_comp._add_output_configure(f'polynomial_control_rates:{name}_rate',
                                                  shape=options['shape'],
                                                  units=get_rate_units(control_units, time_units,
                                                                       deriv=1),
                                                  desc=f'first time-derivative of control {name}')

            phase.connect(src_name=f'polynomial_control_rates:{name}_rate',
                          tgt_name=f'timeseries.all_values:polynomial_control_rates:{name}_rate')

            # Polynomial control second derivatives
            timeseries_comp._add_output_configure(f'polynomial_control_rates:{name}_rate2',
                                                  shape=options['shape'],
                                                  units=get_rate_units(control_units, time_units,
                                                                       deriv=2),
                                                  desc=f'second time-derivative of control {name}')

            phase.connect(src_name=f'polynomial_control_rates:{name}_rate2',
                          tgt_name=f'timeseries.all_values:polynomial_control_rates:{name}_rate2')

        for name, options in phase.parameter_options.items():
            prom_name = f'parameters:{name}'

            if options['include_timeseries']:
                phase.timeseries._add_output_configure(prom_name,
                                                       desc='',
                                                       shape=options['shape'],
                                                       units=options['units'])

                if output_nodes_per_seg is None:
                    src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                else:
                    src_idxs_raw = np.zeros(num_seg * output_nodes_per_seg, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape']).ravel()

                # tgt_name = f'all_values:parameters:{name}'
                # phase.promotes('timeseries', inputs=[(tgt_name, prom_name)],
                #                src_indices=(src_idxs,), flat_src_indices=True)
                # print(src_idxs)
                phase.connect(f'parameter_vals:{name}', f'timeseries.all_values:parameters:{name}',
                              src_indices=src_idxs, flat_src_indices=True)

        for ts_output in phase._timeseries['timeseries']['outputs']:
            var = ts_output['name']
            output_name = ts_output['output_name']
            units = ts_output['units']
            wildcard_units = ts_output['wildcard_units']
            shape = ts_output['shape']

            if '*' in var:  # match outputs from the ODE
                matches = filter(list(ode_outputs.keys()), var)
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

                # Skip the timeseries output if it does not appear to be shaped as a dynamic variable
                # If the full shape does not start with num_nodes, skip this variable.
                if self.is_static_ode_output(v, ode_outputs, self.num_output_nodes):
                    warnings.warn(f'Cannot add ODE output {v} to the timeseries output. It is '
                                  f'sized such that its first dimension != num_nodes.')
                    continue

                shape, units = get_source_metadata(ode_outputs, src=v, user_shape=shape, user_units=units)

                try:
                    timeseries_comp._add_output_configure(output_name, shape=shape, units=units, desc='')
                except ValueError as e:  # OK if it already exists
                    if 'already exists' in str(e):
                        continue
                    else:
                        raise e

                # Failed to find variable, assume it is in the RHS
                phase.connect(src_name=f'ode.{v}',
                              tgt_name=f'timeseries.all_values:{output_name}')

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
            # ode_tgts = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])

            static = options['static_target']
            shape = options['shape']

            # Get connections to each segment
            #
            gd = self.grid_data
            #
            # # We also need to take care of the segments.
            # segs = phase._get_subsystem('segments')

            # for name, options in phase.parameter_options.items():
            #     prom_name = f'parameters:{name}'
            #     shape = options['shape']
            #     units = options['units']
            #
            for i in range(gd.num_segments):
                # seg_comp = segs._get_subsystem(f'segment_{i}')
                # seg_comp.add_input(name=f'parameters:{name}', val=np.ones(shape), units=units,
                #                    desc=f'values of parameter {name}.')
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

    def _get_rate_source_path(self, state_var, nodes, phase):
        """
        Return the rate source location for a given state name.
        Parameters
        ----------
        state_var : str
            Name of the state.
        nodes : str
            The nodes subset which we are connecting from the rate source. Note used in SolveIVP.
        phase : dymos.Phase
            Phase object containing the rate source.
        Returns
        -------
        str
            Path to the rate source.
        np.array
            Source indices for the connection from the rate source to the target.
        """
        var = phase.state_options[state_var]['rate_source']
        node_idxs = None

        if var == 'time':
            rate_path = 'time'
        elif var == 'time_phase':
            rate_path = 'time_phase'
        elif phase.state_options is not None and var in phase.state_options:
            rate_path = f'state_mux_comp.states:{var}'
        elif phase.control_options is not None and var in phase.control_options:
            rate_path = f'control_values:{var}'
        elif phase.polynomial_control_options is not None and var in phase.polynomial_control_options:
            rate_path = f'polynomial_control_values:{var}'
        elif phase.parameter_options is not None and var in phase.parameter_options:
            rate_path = f'parameter_vals:{var}'
            num_seg = self.grid_data.num_segments
            node_idxs = np.zeros(num_seg * self.options['output_nodes_per_seg'], dtype=int)
        elif var.endswith('_rate') and phase.control_options is not None and \
                var[:-5] in phase.control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate2') and phase.control_options is not None and \
                var[:-6] in phase.control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate') and phase.polynomial_control_options is not None and \
                var[:-5] in phase.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        elif var.endswith('_rate2') and phase.polynomial_control_options is not None and \
                var[:-6] in phase.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        else:
            rate_path = f'ode.{var}'

        return rate_path, node_idxs
