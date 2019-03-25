from __future__ import division, print_function, absolute_import

import numpy as np
from dymos.phases.phase_base import PhaseBase

from openmdao.api import Group, IndepVarComp
from six import iteritems

from ..options import TimeOptionsDictionary
from .components.segment_simulation_comp import SegmentSimulationComp
from .components.ode_integration_interface import ODEIntegrationInterface
from .components.segment_state_mux_comp import SegmentStateMuxComp
from .components.solve_ivp_timeseries_comp import SolveIVPTimeseriesOutputComp
from .components.solve_ivp_control_group import SolveIVPControlGroup
from .components.solve_ivp_polynomial_control_group import SolveIVPPolynomialControlGroup

from ..components import TimeComp
from ...utils.misc import get_rate_units
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData


class SolveIVPPhase(PhaseBase):
    """
    SolveIVPPhase provides explicit integration via the scipy.integrate.solve_ivp function.

    SolveIVPPhase is conceptually similar to the RungeKuttaPhase, where the ODE is integrated
    segment to segment.  However, whereas RungeKuttaPhase evaluates the ODE simultaneously across
    segments to quickly solve the integration, SolveIVPPhase utilizes scipy.integrate.solve_ivp
    to accurately propagate each segment with a variable time-step integration routine.

    Currently SolveIVPPhase provides no design variables, constraints, or objectives for
    optimization since we currently don't provide analytic derivatives across the integration steps.

    """
    def __init__(self, from_phase=None, **kwargs):
        super(SolveIVPPhase, self).__init__(**kwargs)

        self._from_phase = from_phase
        """The phase whose results we are simulating explicitly."""

        if from_phase is not None:
            self.options['ode_class'] = from_phase.options['ode_class']
        else:
            self.options['ode_class'] = None

    def initialize(self):
        super(SolveIVPPhase, self).initialize()
        self.options.declare('from_phase', default=None, types=(PhaseBase,), allow_none=True,
                             desc='Phase from which this phase is being instantiated.')

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

    def setup(self):

        if self._from_phase is not None:
            self.user_time_options = TimeOptionsDictionary()
            self.user_time_options.update(self._from_phase.time_options)
            self.user_state_options = self._from_phase.state_options.copy()
            self.user_control_options = self._from_phase.control_options.copy()
            self.user_polynomial_control_options = self._from_phase.polynomial_control_options.copy()
            self.user_design_parameter_options = self._from_phase.design_parameter_options.copy()
            self.user_input_parameter_options = self._from_phase.input_parameter_options.copy()
            self.user_traj_parameter_options = self._from_phase.traj_parameter_options.copy()

            self._timeseries_outputs = self._from_phase._timeseries_outputs.copy()

            self.grid_data = self._from_phase.grid_data
            self.options['num_segments'] = self.grid_data.num_segments
            self.options['transcription_order'] = self.grid_data.transcription_order
            self.options['segment_ends'] = self.grid_data.segment_ends
            self.options['compressed'] = self.grid_data.compressed

        if self.grid_data is None:
            num_seg = self.options['num_segments']
            transcription_order = self.options['transcription_order']
            seg_ends = self.options['segment_ends']
            compressed = self.options['compressed']

            self.grid_data = GridData(num_segments=num_seg,
                                      transcription='gauss-lobatto',
                                      transcription_order=transcription_order,
                                      segment_ends=seg_ends, compressed=compressed)

        super(SolveIVPPhase, self).setup()

    def _setup_time(self):
        time_options = self.time_options
        time_units = time_options['units']
        num_seg = self.options['num_segments']
        grid_data = self.grid_data
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        indeps, externals, comps = super(SolveIVPPhase, self)._setup_time()

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

        self.add_subsystem('time', time_comp, promotes_outputs=['time', 'time_phase'],
                           promotes_inputs=externals)
        comps.append('time')

        for i in range(num_seg):
            self.connect('t_initial', 'segment_{0}.t_initial'.format(i))
            self.connect('t_duration', 'segment_{0}.t_duration'.format(i))
            if output_nodes_per_seg is None:
                i1, i2 = grid_data.subset_segment_indices['all'][i, :]
                src_idxs = grid_data.subset_node_indices['all'][i1:i2]
            else:
                src_idxs = np.arange(i * output_nodes_per_seg, output_nodes_per_seg * (i + 1),
                                     dtype=int)
            self.connect('time', 'segment_{0}.time'.format(i), src_indices=src_idxs)
            self.connect('time_phase', 'segment_{0}.time_phase'.format(i), src_indices=src_idxs)

        if self.time_options['targets']:
            self.connect('time', ['ode.{0}'.format(t) for t in time_options['targets']])

        if self.time_options['time_phase_targets']:
            time_phase_tgts = time_options['time_phase_targets']
            self.connect('time_phase',
                         ['ode.{0}'.format(t) for t in time_phase_tgts])

        if self.time_options['t_initial_targets']:
            time_phase_tgts = time_options['t_initial_targets']
            self.connect('t_initial', ['ode.{0}'.format(t) for t in time_phase_tgts])

        if self.time_options['t_duration_targets']:
            time_phase_tgts = time_options['t_duration_targets']
            self.connect('t_duration',
                         ['ode.{0}'.format(t) for t in time_phase_tgts])

        return comps

    def _setup_rhs(self):

        gd = self.grid_data
        num_seg = gd.num_segments

        self._indep_states_ivc = self.add_subsystem('indep_states', IndepVarComp(),
                                                    promotes_outputs=['*'])

        segments_group = self.add_subsystem(name='segments', subsys=Group(),
                                            promotes_outputs=['*'], promotes_inputs=['*'])

        # All segments use a common ODEIntegrationInterface to save some memory.
        # If this phase is ever converted to a multiple-shooting formulation, this will
        # have to change.
        ode_interface = ODEIntegrationInterface(
            ode_class=self.options['ode_class'],
            time_options=self.time_options,
            state_options=self.state_options,
            control_options=self.control_options,
            polynomial_control_options=self.polynomial_control_options,
            design_parameter_options=self.design_parameter_options,
            input_parameter_options=self.input_parameter_options,
            traj_parameter_options=self.traj_parameter_options,
            ode_init_kwargs=self.options['ode_init_kwargs'])

        for i in range(num_seg):
            seg_i_comp = SegmentSimulationComp(
                index=i,
                method=self.options['method'],
                atol=self.options['atol'],
                rtol=self.options['rtol'],
                grid_data=self.grid_data,
                ode_class=self.options['ode_class'],
                ode_init_kwargs=self.options['ode_init_kwargs'],
                time_options=self.time_options,
                state_options=self.state_options,
                control_options=self.control_options,
                polynomial_control_options=self.polynomial_control_options,
                design_parameter_options=self.design_parameter_options,
                input_parameter_options=self.input_parameter_options,
                traj_parameter_options=self.traj_parameter_options,
                ode_integration_interface=ode_interface,
                output_nodes_per_seg=self.options['output_nodes_per_seg'])

            segments_group.add_subsystem('segment_{0}'.format(i), subsys=seg_i_comp)

        # scipy.integrate.solve_ivp does not actually evaluate the ODE at the desired output points,
        # but just returns the time and interpolated integrated state values there instead. We need
        # to instantiate a second ODE group that will call the ODE at those points so that we can
        # accurately obtain timeseries for ODE outputs.
        self.add_subsystem('state_mux_comp',
                           SegmentStateMuxComp(grid_data=gd, state_options=self.state_options,
                                               output_nodes_per_seg=self.options['output_nodes_per_seg']))

        if self.options['output_nodes_per_seg'] is None:
            num_output_nodes = gd.subset_num_nodes['all']
        else:
            num_output_nodes = num_seg * self.options['output_nodes_per_seg']

        self.add_subsystem('ode', self.options['ode_class'](num_nodes=num_output_nodes,
                                                            **self.options['ode_init_kwargs']))

    def _setup_states(self):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        num_seg = self.grid_data.num_segments

        for state_name, options in iteritems(self.state_options):
            self._indep_states_ivc.add_output('initial_states:{0}'.format(state_name),
                                              val=np.ones(((1,) + options['shape'])),
                                              units=options['units'])

            # Connect the initial state to the first segment
            src_idxs = get_src_indices_by_row([0], options['shape'])

            self.connect('initial_states:{0}'.format(state_name),
                         'segment_0.initial_states:{0}'.format(state_name),
                         src_indices=src_idxs, flat_src_indices=True)

            self.connect('segment_0.states:{0}'.format(state_name),
                         'state_mux_comp.segment_0_states:{0}'.format(state_name))

            if options['targets']:
                self.connect('state_mux_comp.states:{0}'.format(state_name),
                             ['ode.{0}'.format(t) for t in options['targets']])

            # Connect the final state in segment n to the initial state in segment n + 1
            for i in range(1, num_seg):
                if self.options['output_nodes_per_seg'] is None:
                    nnps_i = self.grid_data.subset_num_nodes_per_segment['all'][i]
                else:
                    nnps_i = self.options['output_nodes_per_seg']

                src_idxs = get_src_indices_by_row([nnps_i-1], shape=options['shape'])
                self.connect('segment_{0}.states:{1}'.format(i-1, state_name),
                             'segment_{0}.initial_states:{1}'.format(i, state_name),
                             src_indices=src_idxs, flat_src_indices=True)

                self.connect('segment_{0}.states:{1}'.format(i, state_name),
                             'state_mux_comp.segment_{0}_states:{1}'.format(i, state_name))

    def _setup_controls(self):
        grid_data = self.grid_data
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        self._check_control_options()

        self._check_control_options()

        if self.control_options:
            control_group = SolveIVPControlGroup(control_options=self.control_options,
                                                 time_units=self.time_options['units'],
                                                 grid_data=self.grid_data,
                                                 output_nodes_per_seg=output_nodes_per_seg)

            self.add_subsystem('control_group',
                               subsys=control_group,
                               promotes=['controls:*', 'control_values:*', 'control_values_all:*',
                                         'control_rates:*'])
            self.connect('time.dt_dstau', 'control_group.dt_dstau')

        for name, options in iteritems(self.control_options):
            for i in range(grid_data.num_segments):
                i1, i2 = grid_data.subset_segment_indices['control_disc'][i, :]
                seg_idxs = grid_data.subset_node_indices['control_disc'][i1:i2]
                src_idxs = get_src_indices_by_row(row_idxs=seg_idxs, shape=options['shape'])
                self.connect(src_name='control_values_all:{0}'.format(name),
                             tgt_name='segment_{0}.controls:{1}'.format(i, name),
                             src_indices=src_idxs, flat_src_indices=True)

            if self.control_options[name]['targets']:
                src_name = 'control_values:{0}'.format(name)
                targets = self.control_options[name]['targets']
                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets])

            if self.control_options[name]['rate_targets']:
                src_name = 'control_rates:{0}_rate'.format(name)
                targets = self.control_options[name]['rate_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets])

            if self.control_options[name]['rate2_targets']:
                src_name = 'control_rates:{0}_rate2'.format(name)
                targets = self.control_options[name]['rate2_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets])

    def _setup_polynomial_controls(self):
        if self.polynomial_control_options:
            sys = SolveIVPPolynomialControlGroup(grid_data=self.grid_data,
                                                 polynomial_control_options=self.polynomial_control_options,
                                                 time_units=self.time_options['units'],
                                                 output_nodes_per_seg=self.options['output_nodes_per_seg'])
            self.add_subsystem('polynomial_control_group', subsys=sys,
                               promotes_inputs=['*'], promotes_outputs=['*'])

        for name, options in iteritems(self.polynomial_control_options):

            for iseg in range(self.grid_data.num_segments):
                self.connect(src_name='polynomial_controls:{0}'.format(name),
                             tgt_name='segment_{0}.polynomial_controls:{1}'.format(iseg, name))

            if self.polynomial_control_options[name]['targets']:
                src_name = 'polynomial_control_values:{0}'.format(name)
                targets = self.polynomial_control_options[name]['targets']
                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets])

            if self.polynomial_control_options[name]['rate_targets']:
                src_name = 'polynomial_control_rates:{0}_rate'.format(name)
                targets = self.polynomial_control_options[name]['rate_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets])

            if self.polynomial_control_options[name]['rate2_targets']:
                src_name = 'polynomial_control_rates:{0}_rate2'.format(name)
                targets = self.polynomial_control_options[name]['rate2_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets])

    def _setup_design_parameters(self):
        super(SolveIVPPhase, self)._setup_design_parameters()
        num_seg = self.grid_data.num_segments
        for name, options in iteritems(self.design_parameter_options):
            self.connect('design_parameters:{0}'.format(name),
                         ['segment_{0}.design_parameters:{1}'.format(iseg, name) for iseg in range(num_seg)])

    def _setup_input_parameters(self):
        super(SolveIVPPhase, self)._setup_input_parameters()
        num_seg = self.grid_data.num_segments
        for name, options in iteritems(self.input_parameter_options):
            self.connect('input_parameters:{0}_out'.format(name),
                         ['segment_{0}.input_parameters:{1}'.format(iseg, name) for iseg in range(num_seg)])

    def _setup_traj_input_parameters(self):
        super(SolveIVPPhase, self)._setup_traj_input_parameters()
        num_seg = self.grid_data.num_segments
        for name, options in iteritems(self.traj_parameter_options):
            self.connect('traj_parameters:{0}_out'.format(name),
                         ['segment_{0}.traj_parameters:{1}'.format(iseg, name) for iseg in range(num_seg)])

    def _setup_defects(self):
        """
        Setup the Continuity component as necessary.
        """
        pass

    def _setup_endpoint_conditions(self):
        pass

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        pass

    def _setup_timeseries_outputs(self):
        gd = self.grid_data
        num_seg = gd.num_segments
        time_units = self.time_options['units']
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        timeseries_comp = \
            SolveIVPTimeseriesOutputComp(grid_data=gd,
                                         output_nodes_per_seg=self.options['output_nodes_per_seg'])

        self.add_subsystem('timeseries', subsys=timeseries_comp)

        timeseries_comp._add_timeseries_output('time',
                                               var_class='time',
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
            self.connect(src_name='state_mux_comp.states:{0}'.format(name),
                         tgt_name='timeseries.all_values:states:{0}'.format(name))

        for name, options in iteritems(self.control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)

            self.connect(src_name='control_values:{0}'.format(name),
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

        for name, options in iteritems(self.polynomial_control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['segment_ends']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            self.connect(src_name='polynomial_control_values:{0}'.format(name),
                         tgt_name='timeseries.all_values:polynomial_controls:{0}'.format(name))

            # Polynomial control rates
            timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            self.connect(src_name='polynomial_control_rates:{0}_rate'.format(name),
                         tgt_name='timeseries.all_values:polynomial_control_rates'
                                  ':{0}_rate'.format(name))

            # Polynomial control second derivatives
            timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                   '{0}_rate2'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            self.connect(src_name='polynomial_control_rates:{0}_rate2'.format(name),
                         tgt_name='timeseries.all_values:polynomial_control_rates'
                                  ':{0}_rate2'.format(name))

        for name, options in iteritems(self.design_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            if output_nodes_per_seg is None:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            else:
                src_idxs_raw = np.zeros(num_seg * output_nodes_per_seg, dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='design_parameters:{0}'.format(name),
                         tgt_name='timeseries.all_values:design_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.input_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   units=units)

            if output_nodes_per_seg is None:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            else:
                src_idxs_raw = np.zeros(num_seg * output_nodes_per_seg, dtype=int)

            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='input_parameters:{0}_out'.format(name),
                         tgt_name='timeseries.all_values:input_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.traj_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('traj_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   units=units)

            if output_nodes_per_seg is None:
                src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
            else:
                src_idxs_raw = np.zeros(num_seg * output_nodes_per_seg, dtype=int)

            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='traj_parameters:{0}_out'.format(name),
                         tgt_name='timeseries.all_values:traj_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for var, options in iteritems(self._timeseries_outputs):
            output_name = options['output_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            # Failed to find variable, assume it is in the RHS
            self.connect(src_name='ode.{0}'.format(var),
                         tgt_name='timeseries.all_values:{0}'.format(output_name))

            kwargs = options.copy()
            kwargs.pop('output_name', None)
            timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

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
        num_seg = self.grid_data.num_segments
        output_nodes_per_seg = self.options['output_nodes_per_seg']
        num_final_ode_nodes = self.grid_data.subset_num_nodes['all'] \
            if output_nodes_per_seg is None else num_seg * output_nodes_per_seg

        parameter_options = self.design_parameter_options.copy()
        parameter_options.update(self.input_parameter_options)
        parameter_options.update(self.traj_parameter_options)
        parameter_options.update(self.control_options)

        if name in parameter_options:
            ode_tgts = parameter_options[name]['targets']
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
        # Determine the path to the variable which we will be constraining
        time_units = self.time_options['units']
        var_type = self._classify_var(var)

        if var_type == 'time':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 'time'
        elif var_type == 'time_phase':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 'time_phase'
        elif var_type == 'state':
            state_shape = self.state_options[var]['shape']
            state_units = self.state_options[var]['units']
            shape = state_shape
            units = state_units
            linear = True if loc == 'initial' and self.state_options[var]['fix_initial'] or \
                loc == 'final' and self.state_options[var]['fix_final'] else False
            constraint_path = 'states:{0}'.format(var)
        elif var_type in ('indep_control', 'input_control'):
            control_shape = self.control_options[var]['shape']
            control_units = self.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'control_values:{0}'.format(var)
        elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
            control_shape = self.polynomial_control_options[var]['shape']
            control_units = self.polynomial_control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'polynomial_control_values:{0}'.format(var)
        elif var_type == 'design_parameter':
            control_shape = self.design_parameter_options[var]['shape']
            control_units = self.design_parameter_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'design_parameters:{0}'.format(var)
        elif var_type == 'input_parameter':
            control_shape = self.input_parameter_options[var]['shape']
            control_units = self.input_parameter_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'input_parameters:{0}_out'.format(var)
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5]
            control_shape = self.control_options[control_var]['shape']
            control_units = self.control_options[control_var]['units']
            d = 1 if var_type == 'control_rate' else 2
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5]
            control_shape = self.polynomial_control_options[control_var]['shape']
            control_units = self.polynomial_control_options[control_var]['units']
            d = 1 if var_type == 'polynomial_control_rate' else 2
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'polynomial_control_rates:{0}'.format(var)
        else:
            # Failed to find variable, assume it is in the RHS
            constraint_path = 'ode.{0}'.format(var)
            shape = None
            units = None
            linear = False

        return constraint_path, shape, units, linear

    def initialize_values_from_phase(self, sim_prob):
        """
        Initializes values in the SolveIVPPhase using the phase from which it was created.

        Parameters
        ----------
        sim_prob : Problem
            The problem instance under which the simulation is being performed.
        """
        phs = self._from_phase

        op_dict = dict([(name, options) for (name, options) in phs.list_outputs(units=True,
                                                                                out_stream=None)])
        ip_dict = dict([(name, options) for (name, options) in phs.list_inputs(units=True,
                                                                               out_stream=None)])

        phs_path = phs.pathname + '.' if phs.pathname else ''

        if self.pathname.split('.')[0] == self.name:
            self_path = self.name + '.'
        else:
            self_path = self.pathname.split('.')[0] + '.' + self.name + '.'

        # Set the integration times
        op = op_dict['{0}timeseries.time'.format(phs_path)]
        sim_prob.set_val('{0}t_initial'.format(self_path), op['value'][0, ...])
        sim_prob.set_val('{0}t_duration'.format(self_path), op['value'][-1, ...] - op['value'][0, ...])

        # Assign initial state values
        for name in phs.state_options:
            op = op_dict['{0}timeseries.states:{1}'.format(phs_path, name)]
            sim_prob['{0}initial_states:{1}'.format(self_path, name)][...] = op['value'][0, ...]

        # Assign control values
        for name, options in iteritems(phs.control_options):
            if options['opt']:
                op = op_dict['{0}control_group.indep_controls.controls:{1}'.format(phs_path, name)]
                sim_prob['{0}controls:{1}'.format(self_path, name)][...] = op['value']
            else:
                ip = ip_dict['{0}control_group.control_interp_comp.controls:{1}'.format(phs_path, name)]
                sim_prob['{0}controls:{1}'.format(self_path, name)][...] = ip['value']

        # Assign polynomial control values
        for name, options in iteritems(phs.polynomial_control_options):
            if options['opt']:
                op = op_dict['{0}polynomial_control_group.indep_polynomial_controls.'
                             'polynomial_controls:{1}'.format(phs_path, name)]
                sim_prob['{0}polynomial_controls:{1}'.format(self_path, name)][...] = op['value']
            else:
                ip = ip_dict['{0}polynomial_control_group.interp_comp.'
                             'polynomial_controls:{1}'.format(phs_path, name)]
                sim_prob['{0}polynomial_controls:{1}'.format(self_path, name)][...] = ip['value']

        # Assign design parameter values
        for name in phs.design_parameter_options:
            op = op_dict['{0}design_params.design_parameters:{1}'.format(phs_path, name)]
            sim_prob['{0}design_parameters:{1}'.format(self_path, name)][...] = op['value']

        # Assign input parameter values
        for name in phs.input_parameter_options:
            op = op_dict['{0}input_params.input_parameters:{1}_out'.format(phs_path, name)]
            sim_prob['{0}input_parameters:{1}'.format(self_path, name)][...] = op['value']

        # Assign traj parameter values
        for name in phs.traj_parameter_options:
            op = op_dict['{0}traj_params.traj_parameters:{1}_out'.format(phs_path, name)]
            sim_prob['{0}traj_parameters:{1}'.format(self_path, name)][...] = op['value']
