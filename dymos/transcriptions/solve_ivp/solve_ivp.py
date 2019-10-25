from __future__ import division, print_function, absolute_import

import numpy as np

import openmdao.api as om
from six import iteritems, string_types

from ..transcription_base import TranscriptionBase
from .components import SegmentSimulationComp, ODEIntegrationInterface, SegmentStateMuxComp, \
    SolveIVPControlGroup, SolveIVPPolynomialControlGroup, SolveIVPTimeseriesOutputComp
from ..common import TimeComp
from ...utils.misc import get_rate_units
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

        for i in range(num_seg):
            phase.connect('t_initial', 'segment_{0}.t_initial'.format(i))
            phase.connect('t_duration', 'segment_{0}.t_duration'.format(i))
            if output_nodes_per_seg is None:
                i1, i2 = grid_data.subset_segment_indices['all'][i, :]
                src_idxs = grid_data.subset_node_indices['all'][i1:i2]
            else:
                src_idxs = np.arange(i * output_nodes_per_seg, output_nodes_per_seg * (i + 1),
                                     dtype=int)
            phase.connect('time', 'segment_{0}.time'.format(i), src_indices=src_idxs)
            phase.connect('time_phase', 'segment_{0}.time_phase'.format(i), src_indices=src_idxs)

        if phase.time_options['targets']:
            phase.connect('time', ['ode.{0}'.format(t) for t in time_options['targets']])

        if phase.time_options['time_phase_targets']:
            time_phase_tgts = time_options['time_phase_targets']
            phase.connect('time_phase',
                          ['ode.{0}'.format(t) for t in time_phase_tgts])

        if phase.time_options['t_initial_targets']:
            time_phase_tgts = time_options['t_initial_targets']
            phase.connect('t_initial', ['ode.{0}'.format(t) for t in time_phase_tgts])

        if phase.time_options['t_duration_targets']:
            time_phase_tgts = time_options['t_duration_targets']
            phase.connect('t_duration',
                          ['ode.{0}'.format(t) for t in time_phase_tgts])

    def setup_states(self, phase):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        num_seg = self.grid_data.num_segments

        indep_states_ivc = phase.add_subsystem('indep_states', om.IndepVarComp(),
                                               promotes_outputs=['*'])

        for state_name, options in iteritems(phase.state_options):
            indep_states_ivc.add_output('initial_states:{0}'.format(state_name),
                                        val=np.ones(((1,) + options['shape'])),
                                        units=options['units'])

            # Connect the initial state to the first segment
            src_idxs = get_src_indices_by_row([0], options['shape'])

            phase.connect('initial_states:{0}'.format(state_name),
                          'segment_0.initial_states:{0}'.format(state_name),
                          src_indices=src_idxs, flat_src_indices=True)

            phase.connect('segment_0.states:{0}'.format(state_name),
                          'state_mux_comp.segment_0_states:{0}'.format(state_name))

            if options['targets']:
                phase.connect('state_mux_comp.states:{0}'.format(state_name),
                              ['ode.{0}'.format(t) for t in options['targets']])

            # Connect the final state in segment n to the initial state in segment n + 1
            for i in range(1, num_seg):
                if self.options['output_nodes_per_seg'] is None:
                    nnps_i = self.grid_data.subset_num_nodes_per_segment['all'][i]
                else:
                    nnps_i = self.options['output_nodes_per_seg']

                src_idxs = get_src_indices_by_row([nnps_i-1], shape=options['shape'])
                phase.connect('segment_{0}.states:{1}'.format(i-1, state_name),
                              'segment_{0}.initial_states:{1}'.format(i, state_name),
                              src_indices=src_idxs, flat_src_indices=True)

                phase.connect('segment_{0}.states:{1}'.format(i, state_name),
                              'state_mux_comp.segment_{0}_states:{1}'.format(i, state_name))

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
                design_parameter_options=phase.design_parameter_options,
                input_parameter_options=phase.input_parameter_options,
                output_nodes_per_seg=self.options['output_nodes_per_seg'])

            segments_group.add_subsystem('segment_{0}'.format(i), subsys=seg_i_comp)

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

    def setup_controls(self, phase):
        grid_data = self.grid_data
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
            phase.connect('dt_dstau', 'control_group.dt_dstau')

        for name, options in iteritems(phase.control_options):
            for i in range(grid_data.num_segments):
                i1, i2 = grid_data.subset_segment_indices['control_disc'][i, :]
                seg_idxs = grid_data.subset_node_indices['control_disc'][i1:i2]
                src_idxs = get_src_indices_by_row(row_idxs=seg_idxs, shape=options['shape'])
                phase.connect(src_name='control_values_all:{0}'.format(name),
                              tgt_name='segment_{0}.controls:{1}'.format(i, name),
                              src_indices=src_idxs, flat_src_indices=True)

            if phase.control_options[name]['targets']:
                src_name = 'control_values:{0}'.format(name)
                targets = phase.control_options[name]['targets']
                phase.connect(src_name, ['ode.{0}'.format(t) for t in targets])

            if phase.control_options[name]['rate_targets']:
                src_name = 'control_rates:{0}_rate'.format(name)
                targets = phase.control_options[name]['rate_targets']
                phase.connect(src_name, ['ode.{0}'.format(t) for t in targets])

            if phase.control_options[name]['rate2_targets']:
                src_name = 'control_rates:{0}_rate2'.format(name)
                targets = phase.control_options[name]['rate2_targets']
                phase.connect(src_name, ['ode.{0}'.format(t) for t in targets])

    def setup_polynomial_controls(self, phase):
        if phase.polynomial_control_options:
            sys = SolveIVPPolynomialControlGroup(grid_data=self.grid_data,
                                                 polynomial_control_options=phase.polynomial_control_options,
                                                 time_units=phase.time_options['units'],
                                                 output_nodes_per_seg=self.options['output_nodes_per_seg'])
            phase.add_subsystem('polynomial_control_group', subsys=sys,
                                promotes_inputs=['*'], promotes_outputs=['*'])

        for name, options in iteritems(phase.polynomial_control_options):

            for iseg in range(self.grid_data.num_segments):
                phase.connect(src_name='polynomial_controls:{0}'.format(name),
                              tgt_name='segment_{0}.polynomial_controls:{1}'.format(iseg, name))

            if phase.polynomial_control_options[name]['targets']:
                src_name = 'polynomial_control_values:{0}'.format(name)
                targets = phase.polynomial_control_options[name]['targets']
                if isinstance(targets, string_types):
                    targets = [targets]
                phase.connect(src_name, ['ode.{0}'.format(t) for t in targets])

            if phase.polynomial_control_options[name]['rate_targets']:
                src_name = 'polynomial_control_rates:{0}_rate'.format(name)
                targets = phase.polynomial_control_options[name]['rate_targets']
                if isinstance(targets, string_types):
                    targets = [targets]
                phase.connect(src_name, ['ode.{0}'.format(t) for t in targets])

            if phase.polynomial_control_options[name]['rate2_targets']:
                src_name = 'polynomial_control_rates:{0}_rate2'.format(name)
                targets = phase.polynomial_control_options[name]['rate2_targets']
                if isinstance(targets, string_types):
                    targets = [targets]
                phase.connect(src_name, ['ode.{0}'.format(t) for t in targets])

    def setup_design_parameters(self, phase):
        super(SolveIVP, self).setup_design_parameters(phase)
        num_seg = self.grid_data.num_segments
        for name, options in iteritems(phase.design_parameter_options):
            phase.connect('design_parameters:{0}'.format(name),
                          ['segment_{0}.design_parameters:{1}'.format(iseg, name) for iseg in range(num_seg)])

    def setup_input_parameters(self, phase):
        super(SolveIVP, self).setup_input_parameters(phase)
        num_seg = self.grid_data.num_segments
        for name, options in iteritems(phase.input_parameter_options):
            phase.connect('input_parameters:{0}_out'.format(name),
                          ['segment_{0}.input_parameters:{1}'.format(iseg, name) for iseg in range(num_seg)])

    def setup_defects(self, phase):
        """
        SolveIVP poses no defects.
        """
        pass

    def setup_objective(self, phase):
        pass

    def setup_endpoint_conditions(self, phase):
        pass

    def setup_path_constraints(self, phase):
        pass

    def setup_boundary_constraints(self, loc, phase):
        pass

    def setup_solvers(self, phase):
        pass

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data
        num_seg = gd.num_segments
        time_units = phase.time_options['units']
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        timeseries_comp = \
            SolveIVPTimeseriesOutputComp(input_grid_data=gd,
                                         output_nodes_per_seg=self.options['output_nodes_per_seg'])

        phase.add_subsystem('timeseries', subsys=timeseries_comp)

        timeseries_comp._add_timeseries_output('time',
                                               var_class='time',
                                               units=time_units)

        phase.connect(src_name='time', tgt_name='timeseries.all_values:time')

        timeseries_comp._add_timeseries_output('time_phase',
                                               var_class=phase.classify_var('time_phase'),
                                               units=time_units)
        phase.connect(src_name='time_phase', tgt_name='timeseries.all_values:time_phase')

        for name, options in iteritems(phase.state_options):
            timeseries_comp._add_timeseries_output('states:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=options['units'])
            phase.connect(src_name='state_mux_comp.states:{0}'.format(name),
                          tgt_name='timeseries.all_values:states:{0}'.format(name))

        for name, options in iteritems(phase.control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)

            phase.connect(src_name='control_values:{0}'.format(name),
                          tgt_name='timeseries.all_values:controls:{0}'.format(name))

            # # Control rates
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            phase.connect(src_name='control_rates:{0}_rate'.format(name),
                          tgt_name='timeseries.all_values:control_rates:{0}_rate'.format(name))

            # Control second derivatives
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            phase.connect(src_name='control_rates:{0}_rate2'.format(name),
                          tgt_name='timeseries.all_values:control_rates:{0}_rate2'.format(name))

        for name, options in iteritems(phase.polynomial_control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['segment_ends']
            phase.connect(src_name='polynomial_control_values:{0}'.format(name),
                          tgt_name='timeseries.all_values:polynomial_controls:{0}'.format(name))

            # Polynomial control rates
            timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            phase.connect(src_name='polynomial_control_rates:{0}_rate'.format(name),
                          tgt_name='timeseries.all_values:polynomial_control_rates'
                                   ':{0}_rate'.format(name))

            # Polynomial control second derivatives
            timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                   '{0}_rate2'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            phase.connect(src_name='polynomial_control_rates:{0}_rate2'.format(name),
                          tgt_name='timeseries.all_values:polynomial_control_rates'
                                   ':{0}_rate2'.format(name))

        for name, options in iteritems(phase.design_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            if output_nodes_per_seg is None:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            else:
                src_idxs_raw = np.zeros(num_seg * output_nodes_per_seg, dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='design_parameters:{0}'.format(name),
                          tgt_name='timeseries.all_values:design_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.input_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   units=units)

            if output_nodes_per_seg is None:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            else:
                src_idxs_raw = np.zeros(num_seg * output_nodes_per_seg, dtype=int)

            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='input_parameters:{0}_out'.format(name),
                          tgt_name='timeseries.all_values:input_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for var, options in iteritems(phase._timeseries['timeseries']['outputs']):
            output_name = options['output_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            # Ignore any variables that we've already added (states, times, controls, etc)
            if var_type != 'ode':
                continue

            # Failed to find variable, assume it is in the RHS
            phase.connect(src_name='ode.{0}'.format(var),
                          tgt_name='timeseries.all_values:{0}'.format(output_name))

            kwargs = options.copy()
            kwargs.pop('output_name', None)
            timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

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

        parameter_options = phase.design_parameter_options.copy()
        parameter_options.update(phase.input_parameter_options)
        parameter_options.update(phase.control_options)

        if name in parameter_options:
            ode_tgts = parameter_options[name]['targets']
            if isinstance(ode_tgts, str):
                ode_tgts = [ode_tgts]
            dynamic = parameter_options[name]['dynamic']
            shape = parameter_options[name]['shape']

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

            connection_info.append((['ode.{0}'.format(tgt) for tgt in ode_tgts], src_idxs))

        return connection_info

    def _get_boundary_constraint_src(self, var, loc):
        pass
