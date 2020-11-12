from fnmatch import filter

import numpy as np

import openmdao.api as om

from ..transcription_base import TranscriptionBase
from .components import SegmentSimulationComp, SegmentStateMuxComp, \
    SolveIVPControlGroup, SolveIVPPolynomialControlGroup, SolveIVPTimeseriesOutputComp
from ..common import TimeComp
from ...utils.misc import get_rate_units, get_target_metadata, get_source_metadata, \
    _unspecified
from ...utils.introspection import get_targets
from ...utils.indexing import get_src_indices_by_row


class SolveIVP(TranscriptionBase):
    """
    The SolveIVP Transcription class.

    SolveIVP transcription in Dymos uses the scipy.simulate.solve_ivp method to explicitly integrate
    the states from the phase initial time to the phase final time.

    SolveIVP transcription does not currently support optimization since it does not propagate
    analytic derivatives through the ODE.
    """
    def __init__(self, grid_data=None, **kwargs):
        super(SolveIVP, self).__init__(**kwargs)
        self.grid_data = grid_data
        self._rhs_source = 'ode'

    def initialize(self):
        super(SolveIVP, self).initialize()

        self.options.declare('method', default='RK45', values=('RK45', 'RK23', 'BDF'),
                             desc='The integrator used within scipy.integrate.solve_ivp. Currently '
                                  'supports \'RK45\', \'RK23\', and \'BDF\'.')

        self.options.declare('atol', default=1.0E-6, types=(float,),
                             desc='Absolute tolerance passed to scipy.integrate.solve_ivp.')

        self.options.declare('rtol', default=1.0E-6, types=(float,),
                             desc='Relative tolerance passed to scipy.integrate.solve_ivp.')

        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

    def init_grid(self):
        pass

    def setup_time(self, phase):
        time_options = phase.time_options
        time_units = time_options['units']
        num_seg = self.grid_data.num_segments
        grid_data = self.grid_data
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        time_options['input_initial'] = False  # True can break simulation
        time_options['input_duration'] = False

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
        super(SolveIVP, self).configure_time(phase)
        num_seg = self.grid_data.num_segments
        grid_data = self.grid_data
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        phase.time.configure_io()

        for i in range(num_seg):
            phase.connect('t_initial', f'segment_{i}.t_initial')
            phase.connect('t_duration', f'segment_{i}.t_duration')
            if output_nodes_per_seg is None:
                i1, i2 = grid_data.subset_segment_indices['all'][i, :]
                src_idxs = grid_data.subset_node_indices['all'][i1:i2]
            else:
                src_idxs = np.arange(i * output_nodes_per_seg, output_nodes_per_seg * (i + 1),
                                     dtype=int)
            phase.connect('time', f'segment_{i}.time', src_indices=src_idxs)
            phase.connect('time_phase', f'segment_{i}.time_phase', src_indices=src_idxs)

        options = phase.time_options

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, usr_tgts, dynamic in [('time', options['targets'], True),
                                        ('time_phase', options['time_phase_targets'], True),
                                        ('t_initial', options['t_initial_targets'], False),
                                        ('t_duration', options['t_duration_targets'], False)]:

            targets = get_targets(phase.ode, name=name, user_targets=usr_tgts)
            if targets:
                phase.connect(name, [f'ode.{t}' for t in targets])

    def setup_states(self, phase):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        phase.add_subsystem('indep_states', om.IndepVarComp(),
                            promotes_outputs=['*'])

    def configure_states(self, phase):
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
                          src_indices=src_idxs, flat_src_indices=True)

            phase.connect(f'segment_0.states:{state_name}',
                          f'state_mux_comp.segment_0_states:{state_name}')

            targets = get_targets(ode=phase.ode, name=state_name, user_targets=options['targets'])

            if targets:
                phase.connect(f'state_mux_comp.states:{state_name}',
                              [f'ode.{t}' for t in targets])

            # Connect the final state in segment n to the initial state in segment n + 1
            for i in range(1, num_seg):
                if self.options['output_nodes_per_seg'] is None:
                    nnps_i = self.grid_data.subset_num_nodes_per_segment['all'][i]
                else:
                    nnps_i = self.options['output_nodes_per_seg']

                src_idxs = get_src_indices_by_row([nnps_i-1], shape=options['shape'])
                phase.connect(f'segment_{i - 1}.states:{state_name}',
                              f'segment_{i}.initial_states:{state_name}',
                              src_indices=src_idxs, flat_src_indices=True)

                phase.connect(f'segment_{i}.states:{state_name}',
                              f'state_mux_comp.segment_{i}_states:{state_name}')

    def setup_ode(self, phase):
        gd = self.grid_data
        num_seg = gd.num_segments

        segments_group = phase.add_subsystem(name='segments', subsys=om.Group(),
                                             promotes_outputs=['*'], promotes_inputs=['*'])

        for i in range(num_seg):
            seg_i_comp = SegmentSimulationComp(
                index=i,
                method=self.options['method'],
                atol=self.options['atol'],
                rtol=self.options['rtol'],
                grid_data=self.grid_data,
                ode_class=phase.options['ode_class'],
                ode_init_kwargs=phase.options['ode_init_kwargs'],
                time_options=phase.time_options,
                state_options=phase.state_options,
                control_options=phase.control_options,
                polynomial_control_options=phase.polynomial_control_options,
                parameter_options=phase.parameter_options,
                output_nodes_per_seg=self.options['output_nodes_per_seg'])

            segments_group.add_subsystem(f'segment_{i}', subsys=seg_i_comp)

        # scipy.integrate.solve_ivp does not actually evaluate the ODE at the desired output points,
        # but just returns the time and interpolated integrated state values there instead. We need
        # to instantiate a second ODE group that will call the ODE at those points so that we can
        # accurately obtain timeseries for ODE outputs.
        phase.add_subsystem('state_mux_comp',
                            SegmentStateMuxComp(grid_data=gd, state_options=phase.state_options,
                                                output_nodes_per_seg=self.options['output_nodes_per_seg']))

        if self.options['output_nodes_per_seg'] is None:
            num_output_nodes = gd.subset_num_nodes['all']
        else:
            num_output_nodes = num_seg * self.options['output_nodes_per_seg']

        phase.add_subsystem('ode', phase.options['ode_class'](num_nodes=num_output_nodes,
                                                              **phase.options['ode_init_kwargs']))

    def configure_ode(self, phase):
        gd = self.grid_data
        num_seg = gd.num_segments

        for i in range(num_seg):
            seg_comp = phase.segments._get_subsystem(f'segment_{i}')
            seg_comp.configure_io()

    def setup_controls(self, phase):
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
        ode = phase._get_subsystem(self._rhs_source)

        # Interrogate shapes and units.
        for name, options in phase.control_options.items():

            full_shape, units = get_target_metadata(ode, name=name,
                                                    user_targets=options['targets'],
                                                    user_units=options['units'],
                                                    user_shape=options['shape'],
                                                    control_rate=True)

            if options['units'] is None:
                options['units'] = units

            # Determine and store the pre-discretized state shape for use by other components.
            if len(full_shape) < 2:
                if options['shape'] in (_unspecified, None):
                    options['shape'] = (1, )
            else:
                options['shape'] = full_shape[1:]

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
                              src_indices=src_idxs, flat_src_indices=True)

            targets = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])
            if targets:
                phase.connect(f'control_values:{name}', [f'ode.{t}' for t in targets])

            targets = get_targets(ode=phase.ode, name=f'{name}_rate',
                                  user_targets=options['rate_targets'])
            if targets:
                phase.connect(f'control_rates:{name}_rate',
                              [f'ode.{t}' for t in targets])

            targets = get_targets(ode=phase.ode, name=f'{name}_rate2',
                                  user_targets=options['rate2_targets'])
            if targets:
                phase.connect(f'control_rates:{name}_rate2',
                              [f'ode.{t}' for t in targets])

    def setup_polynomial_controls(self, phase):
        if phase.polynomial_control_options:
            sys = SolveIVPPolynomialControlGroup(grid_data=self.grid_data,
                                                 polynomial_control_options=phase.polynomial_control_options,
                                                 time_units=phase.time_options['units'],
                                                 output_nodes_per_seg=self.options['output_nodes_per_seg'])
            phase.add_subsystem('polynomial_control_group', subsys=sys,
                                promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_polynomial_controls(self, phase):
        # In transcription_base, we get the control units/shape from the target, and then call
        # configure on the control_group.
        super(SolveIVP, self).configure_polynomial_controls(phase)

        # Additional connections.
        for name, options in phase.polynomial_control_options.items():

            for iseg in range(self.grid_data.num_segments):
                phase.connect(src_name=f'polynomial_controls:{name}',
                              tgt_name=f'segment_{iseg}.polynomial_controls:{name}')

            targets = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])
            if targets:
                phase.connect(f'polynomial_control_values:{name}', [f'ode.{t}' for t in targets])

            targets = get_targets(ode=phase.ode, name=f'{name}_rate',
                                  user_targets=options['rate_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate',
                              [f'ode.{t}' for t in targets])

            targets = get_targets(ode=phase.ode, name=f'{name}_rate2',
                                  user_targets=options['rate2_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate2',
                              [f'ode.{t}' for t in targets])

    def configure_parameters(self, phase):
        super(SolveIVP, self).configure_parameters(phase)

        gd = self.grid_data

        # We also need to take care of the segments.
        segs = phase._get_subsystem('segments')

        for name, options in phase.parameter_options.items():
            prom_name = f'parameters:{name}'
            shape, units = get_target_metadata(phase.ode, name=name,
                                               user_targets=options['targets'],
                                               user_shape=options['shape'],
                                               user_units=options['units'])
            options['units'] = units
            options['shape'] = shape

            for i in range(gd.num_segments):
                seg_comp = segs._get_subsystem(f'segment_{i}')
                seg_comp.add_input(name=prom_name, val=np.ones(shape), units=units,
                                   desc=f'values of parameter {name}.')
                segs.promotes(f'segment_{i}', inputs=[prom_name])

    def setup_defects(self, phase):
        """
        SolveIVP poses no defects.
        """
        pass

    def configure_defects(self, phase):
        pass

    def setup_objective(self, phase):
        pass

    def configure_objective(self, phase):
        pass

    def setup_path_constraints(self, phase):
        pass

    def configure_path_constraints(self, phase):
        pass

    def setup_boundary_constraints(self, loc, phase):
        pass

    def configure_boundary_constraints(self, loc, phase):
        pass

    def setup_solvers(self, phase):
        pass

    def configure_solvers(self, phase):
        pass

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data

        timeseries_comp = \
            SolveIVPTimeseriesOutputComp(input_grid_data=gd,
                                         output_nodes_per_seg=self.options['output_nodes_per_seg'])

        phase.add_subsystem('timeseries', subsys=timeseries_comp)

    def configure_timeseries_outputs(self, phase):
        gd = self.grid_data
        num_seg = gd.num_segments
        output_nodes_per_seg = self.options['output_nodes_per_seg']
        time_units = phase.time_options['units']

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

            phase.connect(src_name=self.get_rate_source_path(name, phase),
                          tgt_name=f'timeseries.all_values:state_rates:{name}')

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
                src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                tgt_name = f'all_values:parameters:{name}'
                phase.promotes('timeseries', inputs=[(tgt_name, prom_name)],
                               src_indices=src_idxs, flat_src_indices=True)

        for var, options in phase._timeseries['timeseries']['outputs'].items():
            output_name = options['output_name']
            units = options.get('units', None)
            timeseries_units = options.get('timeseries_units', None)

            if '*' in var:  # match outputs from the ODE
                ode_outputs = {opts['prom_name']: opts for (k, opts) in
                               phase.ode.get_io_metadata(iotypes=('output',)).items()}
                matches = filter(list(ode_outputs.keys()), var)
            else:
                matches = [var]

            for v in matches:
                if '*' in var:
                    output_name = v.split('.')[-1]
                    units = ode_outputs[v]['units']
                    # check for timeseries_units override of ODE units
                    if v in timeseries_units:
                        units = timeseries_units[v]

                # Determine the path to the variable which we will be constraining
                # This is more complicated for path constraints since, for instance,
                # a single state variable has two sources which must be connected to
                # the path component.
                var_type = phase.classify_var(v)

                # Ignore any variables that we've already added (states, times, controls, etc)
                if var_type != 'ode':
                    continue

                shape, units = get_source_metadata(phase.ode, src=v, user_shape=options['shape'],
                                                   user_units=units)

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
        Returns a list containing tuples of each path and related indices to which the
        given parameter name is to be connected.

        Parameters
        ----------
        name : str
            The name of the parameter for which connection information is desired.
        phase
            The phase object to which this transcription applies.

        Returns
        -------
        connection_info : list of (paths, indices)
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
            ode_tgts = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])

            dynamic = options['dynamic']
            shape = options['shape']

            # Connections to the final ODE
            if dynamic:
                src_idxs_raw = np.zeros(num_final_ode_nodes, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                if shape == (1,):
                    src_idxs = src_idxs.ravel()
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                src_idxs = np.squeeze(src_idxs, axis=0)

            connection_info.append(([f'ode.{tgt}' for tgt in ode_tgts], src_idxs))

        return connection_info

    def _get_boundary_constraint_src(self, var, loc):
        pass

    def get_rate_source_path(self, state_var, phase):
        var = phase.state_options[state_var]['rate_source']

        if var == 'time':
            rate_path = 'time'
        elif var == 'time_phase':
            rate_path = 'time_phase'
        elif phase.state_options is not None and var in phase.state_options:
            rate_path = f'state_mux_comp.states:{var}'
        elif phase.control_options is not None and var in phase.control_options:
            rate_path = f'control_values:{var}'
        elif phase.polynomial_control_options is not None and var in phase.polynomial_control_options:
            rate_path = f'polynomial_controls:{var}'
        elif phase.parameter_options is not None and var in phase.parameter_options:
            rate_path = f'parameters:{var}'
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

        return rate_path
