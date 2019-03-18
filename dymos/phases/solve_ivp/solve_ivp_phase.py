from __future__ import division, print_function, absolute_import

from collections import Sequence
import warnings

import numpy as np
from dymos.phases.components import EndpointConditionsComp
from dymos.phases.phase_base import PhaseBase, _unspecified

from openmdao.api import Group, IndepVarComp, NonlinearRunOnce, NonlinearBlockGS, \
    NewtonSolver, BoundsEnforceLS
from six import iteritems

from ..runge_kutta.runge_kutta_phase import RungeKuttaPhase
from ..runge_kutta.components import RungeKuttaStepsizeComp, RungeKuttaStateContinuityIterGroup, \
    RungeKuttaTimeseriesOutputComp, RungeKuttaPathConstraintComp, RungeKuttaControlContinuityComp

from .components.segment_simulation_comp import SegmentSimulationComp
from .components.ode_integration_interface import ODEIntegrationInterface
from .components.segment_state_mux_comp import SegmentStateMuxComp
from ..components import TimeComp
from ...utils.rk_methods import rk_methods
from ...utils.misc import CoerceDesvar, get_rate_units
from ...utils.constants import INF_BOUND
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
    def initialize(self):
        # Note we're calling super on RungeKuttaPhase, and bypassing RungeKuttaPhase's initialize!
        super(SolveIVPPhase, self).initialize()

        self.options.declare('method', default='RK45', values=('RK45', 'RK23', 'BDF'),
                             desc='The integrator used within scipy.integrate.solve_ivp. Currently '
                                  'supports \'RK45\', \'RK23\', and \'BDF\'.')

        self.options.declare('atol', default=1.0E-6, types=(float,),
                             desc='Absolute tolerance passed to scipy.integrate.solve_ivp.')

        self.options.declare('rtol', default=1.0E-6, types=(float,),
                             desc='Relative tolerance passed to scipy.integrate.solve_ivp.')

        self.options.declare('grid_data', default=None, types=(GridData,), allow_none=True,
                             desc='The GridData object containing information on the control '
                                  'parameterization.  Unlike other Phases, SolveIVPPhase should '
                                  'be instantiated with the GridData of whichever phase it is '
                                  'simulating.')

    def setup(self):

        if self.options['grid_data'] is None:
            num_seg = self.options['num_segments']
            transcription_order = self.options['transcription_order']
            seg_ends = self.options['segment_ends']
            compressed = self.options['compressed']

            self.options['grid_data'] = GridData(num_segments=num_seg,
                                                 transcription='gauss-lobatto',
                                                 transcription_order=transcription_order,
                                                 segment_ends=seg_ends, compressed=compressed)

        super(SolveIVPPhase, self).setup()

    def _setup_time(self):
        time_options = self.time_options
        time_units = time_options['units']
        num_seg = self.options['num_segments']
        grid_data = self.options['grid_data']
        num_nodes = grid_data.num_nodes

        indeps, externals, comps = super(SolveIVPPhase, self)._setup_time()

        time_comp = TimeComp(num_nodes=num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau, units=time_units)

        self.add_subsystem('time', time_comp, promotes_outputs=['time', 'time_phase'],
                           promotes_inputs=externals)
        comps.append('time')

        for i in range(num_seg):
            self.connect('t_initial', 'segment_{0}.t_initial'.format(i))
            self.connect('t_duration', 'segment_{0}.t_duration'.format(i))
            i1, i2 = grid_data.subset_segment_indices['all'][i, :]
            src_idxs = grid_data.subset_node_indices['all'][i1:i2]
            self.connect('time', 'segment_{0}.time'.format(i), src_indices=src_idxs)
            self.connect('time_phase', 'segment_{0}.time_phase'.format(i), src_indices=src_idxs)

        if self.time_options['targets']:
            # self.connect('time', ['rk_solve_group.ode.{0}'.format(t) for t in time_tgts],
            #              src_indices=self.grid_data.subset_node_indices['all'])

            self.connect('time', ['ode.{0}'.format(t) for t in time_options['targets']],
                         src_indices=grid_data.subset_node_indices['all'])

        if self.time_options['time_phase_targets']:
            time_phase_tgts = time_options['time_phase_targets']
            # self.connect('time_phase',
            #              ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            self.connect('time_phase',
                         ['ode.{0}'.format(t) for t in time_phase_tgts],
                         src_indices=grid_data.subset_node_indices['all'])

        if self.time_options['t_initial_targets']:
            time_phase_tgts = time_options['t_initial_targets']
            # self.connect('t_initial', ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            self.connect('t_initial', ['ode.{0}'.format(t) for t in time_phase_tgts])

        if self.time_options['t_duration_targets']:
            time_phase_tgts = time_options['t_duration_targets']
            # self.connect('t_duration',
            #              ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            self.connect('t_duration',
                         ['ode.{0}'.format(t) for t in time_phase_tgts])

        return comps

    def _setup_rhs(self):

        gd = self.options['grid_data']
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
                grid_data=self.options['grid_data'],
                ode_class=self.options['ode_class'],
                ode_init_kwargs=self.options['ode_init_kwargs'],
                time_options=self.time_options,
                state_options=self.state_options,
                control_options=self.control_options,
                polynomial_control_options=self.polynomial_control_options,
                design_parameter_options=self.design_parameter_options,
                input_parameter_options=self.input_parameter_options,
                traj_parameter_options=self.traj_parameter_options,
                ode_integration_interface=ode_interface)

            segments_group.add_subsystem('segment_{0}'.format(i), subsys=seg_i_comp)

        # scipy.integrate.solve_ivp does not actually evaluate the ODE at the desired output points,
        # but just returns the time and interpolated integrated state values there instead. We need
        # to instantiate a second ODE group that will call the ODE at those points so that we can
        # accurately obtain timeseries for ODE outputs.
        self.add_subsystem('state_mux_comp',
                           SegmentStateMuxComp(grid_data=gd, state_options=self.state_options))


        self.add_subsystem('ode',
                           self.options['ode_class'](num_nodes=gd.subset_num_nodes['all'],
                                                     **self.options['ode_init_kwargs']))

    def _setup_states(self):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        num_seg = self.options['grid_data'].num_segments

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
                nnps_i = self.options['grid_data'].subset_num_nodes_per_segment['all'][i]
                src_idxs = get_src_indices_by_row([nnps_i-1], shape=options['shape'])
                self.connect('segment_{0}.states:{1}'.format(i-1, state_name),
                             'segment_{0}.initial_states:{1}'.format(i, state_name),
                             src_indices=src_idxs, flat_src_indices=True)

                self.connect('segment_{0}.states:{1}'.format(i, state_name),
                             'state_mux_comp.segment_{0}_states:{1}'.format(i, state_name))

                # self.connect('segment_{0}.states')

            # # Connect the state rate source to the k comp
            # rate_path, src_idxs = self._get_rate_source_path(state_name)
            #
            # self.connect(rate_path,
            #              'rk_solve_group.k_comp.f:{0}'.format(state_name),
            #              src_indices=src_idxs,
            #              flat_src_indices=True)

    def _setup_controls(self):
        super(RungeKuttaPhase, self)._setup_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.control_options):
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            all_idxs = grid_data.subset_node_indices['all']
            segend_src_idxs = get_src_indices_by_row(segment_end_idxs, shape=options['shape'])
            all_src_idxs = get_src_indices_by_row(all_idxs, shape=options['shape'])

            if self.control_options[name]['targets']:
                src_name = 'control_values:{0}'.format(name)
                targets = self.control_options[name]['targets']
                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs.ravel(), flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs.ravel(), flat_src_indices=True)

            if self.control_options[name]['rate_targets']:
                src_name = 'control_rates:{0}_rate'.format(name)
                targets = self.control_options[name]['rate_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs, flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs, flat_src_indices=True)

            if self.control_options[name]['rate2_targets']:
                src_name = 'control_rates:{0}_rate2'.format(name)
                targets = self.control_options[name]['rate2_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs, flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs, flat_src_indices=True)

    def _setup_polynomial_controls(self):
        super(RungeKuttaPhase, self)._setup_polynomial_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.polynomial_control_options):
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            all_idxs = grid_data.subset_node_indices['all']
            segend_src_idxs = get_src_indices_by_row(segment_end_idxs, shape=options['shape'])
            all_src_idxs = get_src_indices_by_row(all_idxs, shape=options['shape'])

            if self.polynomial_control_options[name]['targets']:
                src_name = 'polynomial_control_values:{0}'.format(name)
                targets = self.polynomial_control_options[name]['targets']
                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs.ravel(), flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs.ravel(), flat_src_indices=True)

            if self.polynomial_control_options[name]['rate_targets']:
                src_name = 'polynomial_control_rates:{0}_rate'.format(name)
                targets = self.polynomial_control_options[name]['rate_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs, flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs, flat_src_indices=True)

            if self.polynomial_control_options[name]['rate2_targets']:
                src_name = 'polynomial_control_rates:{0}_rate2'.format(name)
                targets = self.polynomial_control_options[name]['rate2_targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs, flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs, flat_src_indices=True)

    def _setup_defects(self):
        """
        Setup the Continuity component as necessary.
        """
        pass

    def _setup_endpoint_conditions(self):
        pass
    #
    #     jump_comp = self.add_subsystem('indep_jumps', subsys=IndepVarComp(),
    #                                    promotes_outputs=['*'])
    #
    #     jump_comp.add_output('initial_jump:time', val=0.0, units=self.time_options['units'],
    #                          desc='discontinuity in time at the start of the phase')
    #
    #     jump_comp.add_output('final_jump:time', val=0.0, units=self.time_options['units'],
    #                          desc='discontinuity in time at the end of the phase')
    #
    #     ic_comp = EndpointConditionsComp(loc='initial',
    #                                      time_options=self.time_options,
    #                                      state_options=self.state_options,
    #                                      control_options=self.control_options)
    #
    #     self.add_subsystem(name='initial_conditions', subsys=ic_comp, promotes_outputs=['*'])
    #
    #     fc_comp = EndpointConditionsComp(loc='final',
    #                                      time_options=self.time_options,
    #                                      state_options=self.state_options,
    #                                      control_options=self.control_options)
    #
    #     self.add_subsystem(name='final_conditions', subsys=fc_comp, promotes_outputs=['*'])
    #
    #     self.connect('time', 'initial_conditions.initial_value:time')
    #     self.connect('time', 'final_conditions.final_value:time')
    #
    #     self.connect('initial_jump:time',
    #                  'initial_conditions.initial_jump:time')
    #
    #     self.connect('final_jump:time',
    #                  'final_conditions.final_jump:time')
    #
    #     for state_name, options in iteritems(self.state_options):
    #         size = np.prod(options['shape'])
    #         ar = np.arange(size)
    #
    #         jump_comp.add_output('initial_jump:{0}'.format(state_name),
    #                              val=np.zeros(options['shape']),
    #                              units=options['units'],
    #                              desc='discontinuity in {0} at the '
    #                                   'start of the phase'.format(state_name))
    #
    #         jump_comp.add_output('final_jump:{0}'.format(state_name),
    #                              val=np.zeros(options['shape']),
    #                              units=options['units'],
    #                              desc='discontinuity in {0} at the '
    #                                   'end of the phase'.format(state_name))
    #
    #         self.connect('states:{0}'.format(state_name),
    #                      'initial_conditions.initial_value:{0}'.format(state_name))
    #
    #         self.connect('states:{0}'.format(state_name),
    #                      'final_conditions.final_value:{0}'.format(state_name))
    #
    #         self.connect('initial_jump:{0}'.format(state_name),
    #                      'initial_conditions.initial_jump:{0}'.format(state_name),
    #                      src_indices=ar, flat_src_indices=True)
    #
    #         self.connect('final_jump:{0}'.format(state_name),
    #                      'final_conditions.final_jump:{0}'.format(state_name),
    #                      src_indices=ar, flat_src_indices=True)
    #
    #     for control_name, options in iteritems(self.control_options):
    #         size = np.prod(options['shape'])
    #         ar = np.arange(size)
    #
    #         jump_comp.add_output('initial_jump:{0}'.format(control_name),
    #                              val=np.zeros(options['shape']),
    #                              units=options['units'],
    #                              desc='discontinuity in {0} at the '
    #                                   'start of the phase'.format(control_name))
    #
    #         jump_comp.add_output('final_jump:{0}'.format(control_name),
    #                              val=np.zeros(options['shape']),
    #                              units=options['units'],
    #                              desc='discontinuity in {0} at the '
    #                                   'end of the phase'.format(control_name))
    #
    #         self.connect('control_values:{0}'.format(control_name),
    #                      'initial_conditions.initial_value:{0}'.format(control_name))
    #
    #         self.connect('control_values:{0}'.format(control_name),
    #                      'final_conditions.final_value:{0}'.format(control_name))
    #
    #         self.connect('initial_jump:{0}'.format(control_name),
    #                      'initial_conditions.initial_jump:{0}'.format(control_name),
    #                      src_indices=ar, flat_src_indices=True)
    #
    #         self.connect('final_jump:{0}'.format(control_name),
    #                      'final_conditions.final_jump:{0}'.format(control_name),
    #                      src_indices=ar, flat_src_indices=True)

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        pass
        # path_comp = None
        # gd = self.grid_data
        # time_units = self.time_options['units']
        # num_seg = gd.num_segments
        #
        # if self._path_constraints:
        #     path_comp = RungeKuttaPathConstraintComp(grid_data=gd)
        #     self.add_subsystem('path_constraints', subsys=path_comp)
        #
        # for var, options in iteritems(self._path_constraints):
        #     con_units = options.get('units', None)
        #     con_name = options['constraint_name']
        #
        #     # Determine the path to the variable which we will be constraining
        #     # This is more complicated for path constraints since, for instance,
        #     # a single state variable has two sources which must be connected to
        #     # the path component.
        #     var_type = self._classify_var(var)
        #
        #     if var_type == 'time':
        #         options['shape'] = (1,)
        #         options['units'] = time_units if con_units is None else con_units
        #         options['linear'] = True
        #         for iseg in range(gd.num_segments):
        #             self.connect(src_name='time',
        #                          tgt_name='path_constraints.all_values:{0}'.format(con_name),
        #                          src_indices=self.grid_data.subset_node_indices['segment_ends'])
        #     elif var_type == 'time_phase':
        #         options['shape'] = (1,)
        #         options['units'] = time_units if con_units is None else con_units
        #         options['linear'] = True
        #         for iseg in range(gd.num_segments):
        #             self.connect(src_name='time_phase',
        #                          tgt_name='path_constraints.all_values:{0}'.format(con_name),
        #                          src_indices=self.grid_data.subset_node_indices['segment_ends'])
        #     elif var_type == 'state':
        #         state_shape = self.state_options[var]['shape']
        #         state_units = self.state_options[var]['units']
        #         options['shape'] = state_shape
        #         options['units'] = state_units if con_units is None else con_units
        #         options['linear'] = False
        #         row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
        #         row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
        #         src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
        #         self.connect('states:{0}'.format(var),
        #                      'path_constraints.all_values:{0}'.format(var),
        #                      src_indices=src_idxs, flat_src_indices=True)
        #
        #     elif var_type in ('indep_control', 'input_control'):
        #         control_shape = self.control_options[var]['shape']
        #         control_units = self.control_options[var]['units']
        #         options['shape'] = control_shape
        #         options['units'] = control_units if con_units is None else con_units
        #         options['linear'] = True if var_type == 'indep_control' else False
        #
        #         src_rows = self.grid_data.subset_node_indices['segment_ends']
        #         src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])
        #
        #         src = 'control_values:{0}'.format(var)
        #
        #         tgt = 'path_constraints.all_values:{0}'.format(con_name)
        #
        #         self.connect(src_name=src, tgt_name=tgt,
        #                      src_indices=src_idxs, flat_src_indices=True)
        #
        #     elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
        #         control_shape = self.polynomial_control_options[var]['shape']
        #         control_units = self.polynomial_control_options[var]['units']
        #         options['shape'] = control_shape
        #         options['units'] = control_units if con_units is None else con_units
        #         options['linear'] = False
        #
        #         src_rows = self.grid_data.subset_node_indices['segment_ends']
        #         src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])
        #
        #         src = 'polynomial_control_values:{0}'.format(var)
        #
        #         tgt = 'path_constraints.all_values:{0}'.format(con_name)
        #
        #         self.connect(src_name=src, tgt_name=tgt,
        #                      src_indices=src_idxs, flat_src_indices=True)
        #
        #     elif var_type in ('control_rate', 'control_rate2'):
        #         if var.endswith('_rate'):
        #             control_name = var[:-5]
        #         elif var.endswith('_rate2'):
        #             control_name = var[:-6]
        #         control_shape = self.control_options[control_name]['shape']
        #         control_units = self.control_options[control_name]['units']
        #         options['shape'] = control_shape
        #
        #         if var_type == 'control_rate':
        #             options['units'] = get_rate_units(control_units, time_units) \
        #                 if con_units is None else con_units
        #         elif var_type == 'control_rate2':
        #             options['units'] = get_rate_units(control_units, time_units, deriv=2) \
        #                 if con_units is None else con_units
        #
        #         options['linear'] = False
        #
        #         src_rows = self.grid_data.subset_node_indices['segment_ends']
        #         src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])
        #
        #         src = 'control_rates:{0}'.format(var)
        #
        #         tgt = 'path_constraints.all_values:{0}'.format(con_name)
        #
        #         self.connect(src_name=src, tgt_name=tgt,
        #                      src_indices=src_idxs, flat_src_indices=True)
        #
        #     elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
        #         if var.endswith('_rate'):
        #             control_name = var[:-5]
        #         elif var.endswith('_rate2'):
        #             control_name = var[:-6]
        #         control_shape = self.polynomial_control_options[control_name]['shape']
        #         control_units = self.polynomial_control_options[control_name]['units']
        #         options['shape'] = control_shape
        #
        #         if var_type == 'polynomial_control_rate':
        #             options['units'] = get_rate_units(control_units, time_units) \
        #                 if con_units is None else con_units
        #         elif var_type == 'polynomial_control_rate2':
        #             options['units'] = get_rate_units(control_units, time_units, deriv=2) \
        #                 if con_units is None else con_units
        #
        #         options['linear'] = False
        #
        #         src_rows = self.grid_data.subset_node_indices['segment_ends']
        #         src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])
        #
        #         src = 'polynomial_control_rates:{0}'.format(var)
        #
        #         tgt = 'path_constraints.all_values:{0}'.format(con_name)
        #
        #         self.connect(src_name=src, tgt_name=tgt,
        #                      src_indices=src_idxs, flat_src_indices=True)
        #
        #     else:
        #         # Failed to find variable, assume it is in the ODE
        #         options['linear'] = False
        #         if options['shape'] is None:
        #             warnings.warn('Unable to infer shape of path constraint {0}. Assuming scalar.\n'
        #                           'In Dymos 1.0 the shape of ODE outputs must be explictly provided'
        #                           ' via the add_path_constraint method.', DeprecationWarning)
        #             options['shape'] = (1,)
        #
        #         src_rows = np.arange(num_seg * 2, dtype=int)
        #         src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])
        #
        #         src = 'ode.{0}'.format(var)
        #         tgt = 'path_constraints.all_values:{0}'.format(con_name)
        #
        #         self.connect(src_name=src, tgt_name=tgt,
        #                      src_indices=src_idxs, flat_src_indices=True)
        #
        #     kwargs = options.copy()
        #     kwargs.pop('constraint_name', None)
        #     path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def _setup_timeseries_outputs(self):
        return

        gd = self.options['grid_data']
        num_seg = gd.num_segments
        time_units = self.time_options['units']
        timeseries_comp = RungeKuttaTimeseriesOutputComp(grid_data=gd)
        self.add_subsystem('timeseries', subsys=timeseries_comp)
        src_idxs = get_src_indices_by_row(gd.subset_node_indices['segment_ends'], (1,))

        timeseries_comp._add_timeseries_output('time',
                                               var_class=self._classify_var('time'),
                                               units=time_units)
        self.connect(src_name='time', tgt_name='timeseries.segend_values:time',
                     src_indices=src_idxs, flat_src_indices=True)

        timeseries_comp._add_timeseries_output('time_phase',
                                               var_class=self._classify_var('time_phase'),
                                               units=time_units)
        self.connect(src_name='time_phase', tgt_name='timeseries.segend_values:time_phase',
                     src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.state_options):
            timeseries_comp._add_timeseries_output('states:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=options['units'])
            row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
            row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
            src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
            self.connect(src_name='states:{0}'.format(name),
                         tgt_name='timeseries.segend_values:states:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['segment_ends']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            self.connect(src_name='control_values:{0}'.format(name),
                         tgt_name='timeseries.segend_values:controls:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

            # # Control rates
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            self.connect(src_name='control_rates:{0}_rate'.format(name),
                         tgt_name='timeseries.segend_values:control_rates:{0}_rate'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

            # Control second derivatives
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            self.connect(src_name='control_rates:{0}_rate2'.format(name),
                         tgt_name='timeseries.segend_values:control_rates:{0}_rate2'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.polynomial_control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['segment_ends']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            self.connect(src_name='polynomial_control_values:{0}'.format(name),
                         tgt_name='timeseries.segend_values:polynomial_controls:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

            # # Control rates
            timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            self.connect(src_name='polynomial_control_rates:{0}_rate'.format(name),
                         tgt_name='timeseries.segend_values:polynomial_control_rates'
                                  ':{0}_rate'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

            # Control second derivatives
            timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                   '{0}_rate2'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            self.connect(src_name='polynomial_control_rates:{0}_rate2'.format(name),
                         tgt_name='timeseries.segend_values:polynomial_control_rates'
                                  ':{0}_rate2'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.design_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='design_parameters:{0}'.format(name),
                         tgt_name='timeseries.segend_values:design_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.input_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='input_parameters:{0}_out'.format(name),
                         tgt_name='timeseries.segend_values:input_parameters:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(self.traj_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('traj_parameters:{0}'.format(name),
                                                   var_class=self._classify_var(name),
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            self.connect(src_name='traj_parameters:{0}_out'.format(name),
                         tgt_name='timeseries.segend_values:traj_parameters:{0}'.format(name),
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
                         tgt_name='timeseries.segend_values:{0}'.format(output_name))

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
        num_stages = rk_methods[self.options['method']]['num_stages']
        num_iter_ode_nodes = num_seg * num_stages
        num_final_ode_nodes = 2 * num_seg

        parameter_options = self.design_parameter_options.copy()
        parameter_options.update(self.input_parameter_options)
        parameter_options.update(self.traj_parameter_options)
        parameter_options.update(self.control_options)

        if name in parameter_options:
            ode_tgts = parameter_options[name]['targets']
            dynamic = parameter_options[name]['dynamic']
            shape = parameter_options[name]['shape']

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

            if dynamic:
                src_idxs_raw = np.zeros(num_iter_ode_nodes, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                if shape == (1,):
                    src_idxs = src_idxs.ravel()
            else:
                src_idxs_raw = np.zeros(1, dtype=int)
                src_idxs = get_src_indices_by_row(src_idxs_raw, shape)
                src_idxs = np.squeeze(src_idxs, axis=0)

            connection_info.append((['rk_solve_group.ode.{0}'.format(tgt) for tgt in ode_tgts],
                                    src_idxs))

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
