from __future__ import division, print_function, absolute_import

from collections import Iterable
import warnings

import numpy as np
from dymos.phases.components import EndpointConditionsComp
from dymos.phases.phase_base import PhaseBase, _unspecified

from openmdao.api import IndepVarComp, NonlinearRunOnce, NonlinearBlockGS, \
    NewtonSolver, BoundsEnforceLS
from six import iteritems

from .components import RungeKuttaStepsizeComp, RungeKuttaStateContinuityIterGroup, \
    RungeKuttaTimeseriesOutputComp, RungeKuttaPathConstraintComp, RungeKuttaControlContinuityComp
from ..components import TimeComp
from ...utils.rk_methods import rk_methods
from ...utils.misc import CoerceDesvar, get_rate_units
from ...utils.constants import INF_BOUND
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData


class RungeKuttaPhase(PhaseBase):
    """
    RungeKuttaPhase provides explicitly integrated phases where each segment is assumed to be
    a single RK timestep.

    Attributes
    ----------
    self.time_options : dict of TimeOptionsDictionary
        A dictionary of options for time (integration variable) in the phase.

    self.state_options : dict of StateOptionsDictionary
        A dictionary of options for the RHS states in the Phase.

    self.control_options : dict of ControlOptionsDictionary
        A dictionary of options for the controls in the Phase.

    self._ode_controls : dict of ControlOptionsDictionary
        A dictionary of the default options for controllable inputs of the Phase RHS

    """
    def initialize(self):
        super(RungeKuttaPhase, self).initialize()
        self.options['transcription'] = 'explicit'
        self.options.declare('num_segments', types=(int,),
                             desc='The number of segments in the Phase.  In RungeKuttaPhase, each'
                                  'segment is a single timestep of the integration scheme.')

        self.options.declare('segment_ends', default=None, types=(int, Iterable), allow_none=True,
                             desc='The relative endpoint locations for each segment. Must be of '
                                  'length (num_segments + 1).')

        self.options.declare('method', default='rk4', values=('rk4',),
                             desc='The integrator used within the explicit phase.')

        self.options.declare('k_solver_class', default=NonlinearBlockGS,
                             values=(NonlinearBlockGS, NewtonSolver, NonlinearRunOnce),
                             allow_none=True,
                             desc='The nonlinear solver class used to converge the numerical '
                                  'integration across each segment.')

        self.options.declare('k_solver_options', default={'iprint': -1}, types=(dict,),
                             desc='The options passed to the nonlinear solver used to converge the'
                                  'Runge-Kutta propagation across each step.')

    def setup(self):
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription='runge-kutta',
                                  transcription_order=self.options['method'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

        super(RungeKuttaPhase, self).setup()

        #
        # Add a newton solver to converge the continuity between the segments/steps
        #
        self.nonlinear_solver = NewtonSolver()
        self.nonlinear_solver.options['iprint'] = -1
        self.nonlinear_solver.options['solve_subsystems'] = True
        self.nonlinear_solver.options['err_on_maxiter'] = True
        self.nonlinear_solver.linesearch = BoundsEnforceLS()

    def _setup_time(self):
        time_units = self.time_options['units']
        num_seg = self.options['num_segments']
        rk_data = rk_methods[self.options['method']]
        num_nodes = num_seg * rk_data['num_stages']
        grid_data = self.grid_data

        indeps, externals, comps = super(RungeKuttaPhase, self)._setup_time()

        time_comp = TimeComp(num_nodes=num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau, units=time_units)

        self.add_subsystem('time', time_comp, promotes_outputs=['time', 'time_phase'],
                           promotes_inputs=externals)
        comps.append('time')

        h_comp = RungeKuttaStepsizeComp(num_segments=num_seg,
                                        seg_rel_lengths=np.diff(grid_data.segment_ends),
                                        time_units=time_units)

        self.add_subsystem('stepsize_comp', h_comp,
                           promotes_inputs=['t_duration'],
                           promotes_outputs=['h'])

        if self.time_options['targets']:
            time_tgts = self.time_options['targets']

            self.connect('time', ['rk_solve_group.ode.{0}'.format(t) for t in time_tgts],
                         src_indices=self.grid_data.subset_node_indices['all'])

            self.connect('time', ['ode.{0}'.format(t) for t in time_tgts],
                         src_indices=self.grid_data.subset_node_indices['segment_ends'])

        if self.time_options['time_phase_targets']:
            time_phase_tgts = self.time_options['time_phase_targets']
            self.connect('time_phase',
                         ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            self.connect('time_phase',
                         ['ode.{0}'.format(t) for t in time_phase_tgts],
                         src_indices=self.grid_data.subset_node_indices['segment_ends'])

        if self.time_options['t_initial_targets']:
            time_phase_tgts = self.time_options['t_initial_targets']
            self.connect('t_initial', ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            self.connect('t_initial', ['ode.{0}'.format(t) for t in time_phase_tgts])

        if self.time_options['t_duration_targets']:
            time_phase_tgts = self.time_options['t_duration_targets']
            self.connect('t_duration',
                         ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            self.connect('t_duration',
                         ['ode.{0}'.format(t) for t in time_phase_tgts])

        return comps

    def _setup_rhs(self):

        num_connected = len([s for s in self.state_options
                             if self.state_options[s]['connected_initial']])
        promoted_inputs = ['h'] if num_connected == 0 else ['h', 'initial_states:*']

        self.add_subsystem('rk_solve_group',
                           RungeKuttaStateContinuityIterGroup(
                               num_segments=self.options['num_segments'],
                               method=self.options['method'],
                               state_options=self.state_options,
                               time_units=self.time_options['units'],
                               ode_class=self.options['ode_class'],
                               ode_init_kwargs=self.options['ode_init_kwargs'],
                               k_solver_class=self.options['k_solver_class'],
                               k_solver_options=self.options['k_solver_options']),
                           promotes_inputs=promoted_inputs,
                           promotes_outputs=['states:*'])

        # Since the RK Solve group evaluates the ODE at *predicted* state values, we need
        # to instantiate a second ODE group that will call the ODE at the actual integrated
        # state values so that we can accurate evaluate path and boundary constraints and
        # obtain timeseries for ODE outputs.
        self.add_subsystem('ode',
                           self.options['ode_class'](num_nodes=2*self.options['num_segments'],
                                                     **self.options['ode_init_kwargs']))

    def _get_rate_source_path(self, state_name, nodes=None, **kwargs):
        var = self.state_options[state_name]['rate_source']
        shape = self.state_options[state_name]['shape']
        var_type = self._classify_var(var)
        num_segments = self.options['num_segments']
        num_stages = rk_methods[self.options['method']]['num_stages']

        # Determine the path to the variable
        if var_type == 'time':
            rate_path = 'time'
            src_idxs = None
        elif var_type == 'time_phase':
            rate_path = 'time_phase'
            src_idxs = None
        elif var_type == 'state':
            rate_path = 'predicted_states:{0}'.format(var)
        elif var_type == 'indep_control':
            rate_path = 'control_values:{0}'.format(var)
            src_idxs = None
        elif var_type == 'input_control':
            rate_path = 'control_values:{0}'.format(var)
            src_idxs = None
        elif var_type == 'control_rate':
            rate_path = 'control_rates:{0}'.format(var)
            src_idxs = None
        elif var_type == 'control_rate2':
            rate_path = 'control_rates:{0}'.format(var)
            src_idxs = None
        elif var_type == 'design_parameter':
            rate_path = 'design_parameters:{0}'.format(var)
            size = np.prod(self.design_parameter_options[var]['shape'])
            src_idxs = np.zeros(num_segments * num_stages * size, dtype=int)
        elif var_type == 'input_parameter':
            rate_path = 'input_parameters:{0}_out'.format(var)
            size = np.prod(self.input_parameter_options[var]['shape'])
            src_idxs = np.zeros(num_segments * num_stages * size, dtype=int)
        else:
            # Failed to find variable, assume it is in the ODE
            rate_path = 'rk_solve_group.ode.{0}'.format(var)
            state_size = np.prod(shape)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))

        return rate_path, src_idxs

    def _setup_states(self):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        num_seg = self.options['num_segments']
        num_state_input_nodes = num_seg + 1

        for state_name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])

            # Connect the states at the segment ends to the final ODE instance.
            row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
            row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
            src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
            self.connect('states:{0}'.format(state_name),
                         ['ode.{0}'.format(tgt) for tgt in options['targets']],
                         src_indices=src_idxs.ravel(), flat_src_indices=True)

            # Connect the state rate source to the k comp
            rate_path, src_idxs = self._get_rate_source_path(state_name)

            self.connect(rate_path,
                         'rk_solve_group.k_comp.f:{0}'.format(state_name),
                         src_indices=src_idxs,
                         flat_src_indices=True)

            if options['opt']:
                # Set the desvar indices accordingly
                desvar_indices = list(range(size))

                if options['fix_initial']:
                    if options['initial_bounds'] is not None:
                        raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                         'initial_bounds for state {0}'.format(state_name))
                    if options['connected_initial']:
                        raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                         '\'connected_initial=True\' for state {0} in '
                                         'phase {1}.'.format(state_name, self.name))
                    del desvar_indices[:size]

                if options['fix_final']:
                    raise ValueError('Cannot specify \'fix_final=True\' in '
                                     'RungeKuttaPhase'.format(state_name))

                if options['final_bounds'] is not None:
                    raise ValueError('Cannot specify \'final_bounds\' in RungeKuttaPhase '
                                     '(state {0})'.format(state_name))

                if len(desvar_indices) > 0:
                    coerce_desvar_option = CoerceDesvar(num_state_input_nodes, desvar_indices,
                                                        options)

                    lb = np.zeros_like(desvar_indices, dtype=float)
                    lb[:] = -INF_BOUND if coerce_desvar_option('lower') is None else \
                        coerce_desvar_option('lower')

                    ub = np.zeros_like(desvar_indices, dtype=float)
                    ub[:] = INF_BOUND if coerce_desvar_option('upper') is None else \
                        coerce_desvar_option('upper')

                    if options['initial_bounds'] is not None:
                        lb[0] = options['initial_bounds'][0]
                        ub[0] = options['initial_bounds'][-1]

                    self.add_design_var(name='states:{0}'.format(state_name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar_option('scaler'),
                                        adder=coerce_desvar_option('adder'),
                                        ref0=coerce_desvar_option('ref0'),
                                        ref=coerce_desvar_option('ref'),
                                        indices=desvar_indices)

    def _setup_controls(self):
        super(RungeKuttaPhase, self)._setup_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.control_options):
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            all_idxs = grid_data.subset_node_indices['all']
            segend_src_idxs = get_src_indices_by_row(segment_end_idxs, shape=options['shape'])
            all_src_idxs = get_src_indices_by_row(all_idxs, shape=options['shape'])

            if name in self.ode_options._parameters:
                src_name = 'control_values:{0}'.format(name)
                targets = self.ode_options._parameters[name]['targets']
                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs.ravel(), flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs.ravel(), flat_src_indices=True)

            if options['rate_param']:
                src_name = 'control_rates:{0}_rate'.format(name)
                targets = self.ode_options._parameters[options['rate_param']]['targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs, flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs, flat_src_indices=True)

            if options['rate2_param']:
                src_name = 'control_rates:{0}_rate2'.format(name)
                targets = self.ode_options._parameters[options['rate2_param']]['targets']

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

            if name in self.ode_options._parameters:
                src_name = 'polynomial_control_values:{0}'.format(name)
                targets = self.ode_options._parameters[name]['targets']
                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs.ravel(), flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs.ravel(), flat_src_indices=True)

            if options['rate_param']:
                src_name = 'polynomial_control_rates:{0}_rate'.format(name)
                targets = self.ode_options._parameters[options['rate_param']]['targets']

                self.connect(src_name,
                             ['ode.{0}'.format(t) for t in targets],
                             src_indices=segend_src_idxs, flat_src_indices=True)

                self.connect(src_name,
                             ['rk_solve_group.{0}'.format(t) for t in targets],
                             src_indices=all_src_idxs, flat_src_indices=True)

            if options['rate2_param']:
                src_name = 'polynomial_control_rates:{0}_rate2'.format(name)
                targets = self.ode_options._parameters[options['rate2_param']]['targets']

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
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        # Add the continuity constraint component if necessary
        if num_seg > 1 and self.control_options:
            time_units = self.time_options['units']

            self.add_subsystem('continuity_comp',
                               RungeKuttaControlContinuityComp(grid_data=grid_data,
                                                               state_options=self.state_options,
                                                               control_options=self.control_options,
                                                               time_units=time_units),
                               promotes_inputs=['t_duration'])

            for name, options in iteritems(self.control_options):
                # The sub-indices of control_disc indices that are segment ends
                segment_end_idxs = grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(segment_end_idxs, options['shape'], flat=True)

                self.connect('control_values:{0}'.format(name),
                             'continuity_comp.controls:{0}'.format(name),
                             src_indices=src_idxs, flat_src_indices=True)

                self.connect('control_rates:{0}_rate'.format(name),
                             'continuity_comp.control_rates:{}_rate'.format(name),
                             src_indices=src_idxs, flat_src_indices=True)

                self.connect('control_rates:{0}_rate2'.format(name),
                             'continuity_comp.control_rates:{}_rate2'.format(name),
                             src_indices=src_idxs, flat_src_indices=True)

    def _setup_endpoint_conditions(self):

        jump_comp = self.add_subsystem('indep_jumps', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        jump_comp.add_output('initial_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the start of the phase')

        jump_comp.add_output('final_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the end of the phase')

        ic_comp = EndpointConditionsComp(loc='initial',
                                         time_options=self.time_options,
                                         state_options=self.state_options,
                                         control_options=self.control_options)

        self.add_subsystem(name='initial_conditions', subsys=ic_comp, promotes_outputs=['*'])

        fc_comp = EndpointConditionsComp(loc='final',
                                         time_options=self.time_options,
                                         state_options=self.state_options,
                                         control_options=self.control_options)

        self.add_subsystem(name='final_conditions', subsys=fc_comp, promotes_outputs=['*'])

        self.connect('time', 'initial_conditions.initial_value:time')
        self.connect('time', 'final_conditions.final_value:time')

        self.connect('initial_jump:time',
                     'initial_conditions.initial_jump:time')

        self.connect('final_jump:time',
                     'final_conditions.final_jump:time')

        for state_name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            jump_comp.add_output('initial_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(state_name))

            jump_comp.add_output('final_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(state_name))

            self.connect('states:{0}'.format(state_name),
                         'initial_conditions.initial_value:{0}'.format(state_name))

            self.connect('states:{0}'.format(state_name),
                         'final_conditions.final_value:{0}'.format(state_name))

            self.connect('initial_jump:{0}'.format(state_name),
                         'initial_conditions.initial_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(state_name),
                         'final_conditions.final_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

        for control_name, options in iteritems(self.control_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            jump_comp.add_output('initial_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(control_name))

            jump_comp.add_output('final_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(control_name))

            self.connect('control_values:{0}'.format(control_name),
                         'initial_conditions.initial_value:{0}'.format(control_name))

            self.connect('control_values:{0}'.format(control_name),
                         'final_conditions.final_value:{0}'.format(control_name))

            self.connect('initial_jump:{0}'.format(control_name),
                         'initial_conditions.initial_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(control_name),
                         'final_conditions.final_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        path_comp = None
        gd = self.grid_data
        time_units = self.time_options['units']
        num_seg = gd.num_segments

        if self._path_constraints:
            path_comp = RungeKuttaPathConstraintComp(grid_data=gd)
            self.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(self._path_constraints):
            con_units = options.get('units', None)
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                for iseg in range(gd.num_segments):
                    self.connect(src_name='time',
                                 tgt_name='path_constraints.all_values:{0}'.format(con_name),
                                 src_indices=self.grid_data.subset_node_indices['segment_ends'])
            elif var_type == 'time_phase':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                for iseg in range(gd.num_segments):
                    self.connect(src_name='time_phase',
                                 tgt_name='path_constraints.all_values:{0}'.format(con_name),
                                 src_indices=self.grid_data.subset_node_indices['segment_ends'])
            elif var_type == 'state':
                state_shape = self.state_options[var]['shape']
                state_units = self.state_options[var]['units']
                options['shape'] = state_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = False
                row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
                row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
                src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
                self.connect('states:{0}'.format(var),
                             'path_constraints.all_values:{0}'.format(var),
                             src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('indep_control', 'input_control'):
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True if var_type == 'indep_control' else False

                src_rows = self.grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'control_values:{0}'.format(var)

                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                self.connect(src_name=src, tgt_name=tgt,
                             src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
                control_shape = self.polynomial_control_options[var]['shape']
                control_units = self.polynomial_control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = False

                src_rows = self.grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'polynomial_control_values:{0}'.format(var)

                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                self.connect(src_name=src, tgt_name=tgt,
                             src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('control_rate', 'control_rate2'):
                if var.endswith('_rate'):
                    control_name = var[:-5]
                elif var.endswith('_rate2'):
                    control_name = var[:-6]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape

                if var_type == 'control_rate':
                    options['units'] = get_rate_units(control_units, time_units) \
                        if con_units is None else con_units
                elif var_type == 'control_rate2':
                    options['units'] = get_rate_units(control_units, time_units, deriv=2) \
                        if con_units is None else con_units

                options['linear'] = False

                src_rows = self.grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'control_rates:{0}'.format(var)

                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                self.connect(src_name=src, tgt_name=tgt,
                             src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
                if var.endswith('_rate'):
                    control_name = var[:-5]
                elif var.endswith('_rate2'):
                    control_name = var[:-6]
                control_shape = self.polynomial_control_options[control_name]['shape']
                control_units = self.polynomial_control_options[control_name]['units']
                options['shape'] = control_shape

                if var_type == 'polynomial_control_rate':
                    options['units'] = get_rate_units(control_units, time_units) \
                        if con_units is None else con_units
                elif var_type == 'polynomial_control_rate2':
                    options['units'] = get_rate_units(control_units, time_units, deriv=2) \
                        if con_units is None else con_units

                options['linear'] = False

                src_rows = self.grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'polynomial_control_rates:{0}'.format(var)

                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                self.connect(src_name=src, tgt_name=tgt,
                             src_indices=src_idxs, flat_src_indices=True)

            else:
                # Failed to find variable, assume it is in the ODE
                options['linear'] = False
                if options['shape'] is None:
                    warnings.warn('Unable to infer shape of path constraint {0}. Assuming scalar.\n'
                                  'In Dymos 1.0 the shape of ODE outputs must be explictly provided'
                                  ' via the add_path_constraint method.', DeprecationWarning)
                    options['shape'] = (1,)

                src_rows = np.arange(num_seg * 2, dtype=int)
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'ode.{0}'.format(var)
                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                self.connect(src_name=src, tgt_name=tgt,
                             src_indices=src_idxs, flat_src_indices=True)

            kwargs = options.copy()
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def _setup_timeseries_outputs(self):

        gd = self.grid_data
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

        if name in self.ode_options._parameters:
            ode_tgts = self.ode_options._parameters[name]['targets']
            dynamic = self.ode_options._parameters[name]['dynamic']
            shape = self.ode_options._parameters[name]['shape']

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

    def set_state_options(self, name, units=_unspecified, val=1.0,
                          fix_initial=False, fix_final=False, initial_bounds=None,
                          final_bounds=None, lower=None, upper=None, scaler=None, adder=None,
                          ref=None, ref0=None, defect_scaler=1.0, defect_ref=None,
                          connected_initial=False):
        """
        Set options that apply the EOM state variable of the given name.

        Parameters
        ----------
        name : str
            Name of the state variable in the RHS.
        units : str or None
            Units in which the state variable is defined.  Internally components may use different
            units for the state variable, but the IndepVarComp which provides its value will provide
            it in these units, and collocation defects will use these units.  If units is not
            specified here then the value as defined in the ODEOptions (@declare_state) will be
            used.
        val :  ndarray
            The default value of the state at the state discretization nodes of the phase.
        fix_initial : bool(False)
            If True, omit the first value of the state from the design variables (prevent the
            optimizer from changing it).
        fix_final : bool(False)
            If True, omit the final value of the state from the design variables (prevent the
            optimizer from changing it).
        lower : float or ndarray or None (None)
            The lower bound of the state at the nodes of the phase.
        upper : float or ndarray or None (None)
            The upper bound of the state at the nodes of the phase.
        scaler : float or ndarray or None (None)
            The scaler of the state value at the nodes of the phase.
        adder : float or ndarray or None (None)
            The adder of the state value at the nodes of the phase.
        ref0 : float or ndarray or None (None)
            The zero-reference value of the state at the nodes of the phase.
        ref : float or ndarray or None (None)
            The unit-reference value of the state at the nodes of the phase
        defect_scaler : float or ndarray (1.0)
            The scaler of the state defect at the collocation nodes of the phase.
        defect_ref : float or ndarray (1.0)
            The unit-reference value of the state defect at the collocation nodes of the phase. If
            provided, this value overrides defect_scaler.
        connected_initial : bool
            If True, then the initial value for this state comes from an externally connected
            source.

        """
        super(RungeKuttaPhase, self).set_state_options(name=name,
                                                       units=units,
                                                       val=val,
                                                       fix_initial=fix_initial,
                                                       fix_final=fix_final,
                                                       initial_bounds=initial_bounds,
                                                       final_bounds=final_bounds,
                                                       lower=lower,
                                                       upper=upper,
                                                       scaler=scaler,
                                                       adder=adder,
                                                       ref=ref,
                                                       ref0=ref0,
                                                       defect_scaler=defect_scaler,
                                                       defect_ref=defect_ref,
                                                       connected_initial=connected_initial)

    def add_objective(self, name, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False):
        """
        Allows the user to add an objective in the phase.  If name is not a state,
        control, control rate, or 'time', then this is assumed to be the path of the variable
        to be constrained in the RHS.

        Parameters
        ----------
        name : str
            Name of the objective variable.  This should be one of 'time', a state or control
            variable, or the path to an output from the top level of the RHS.
        loc : str
            Where in the phase the objective is to be evaluated.  Valid
            options are 'initial' and 'final'.  The default is 'final'.
        index : int, optional
            If variable is an array at each point in time, this indicates which index is to be
            used as the objective, assuming C-ordered flattening.
        shape : int, optional
            The shape of the objective variable, at a point in time
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        """
        var_type = self._classify_var(name)

        # Determine the path to the variable
        if var_type == 'time':
            obj_path = 'time'
        elif var_type == 'time_phase':
            obj_path = 'time_phase'
        elif var_type == 'state':
            obj_path = 'timeseries.states:{0}'.format(name)
        elif var_type == 'indep_control':
            obj_path = 'control_values:{0}'.format(name)
        elif var_type == 'input_control':
            obj_path = 'control_values:{0}'.format(name)
        elif var_type == 'control_rate':
            control_name = name[:-5]
            obj_path = 'control_rates:{0}_rate'.format(control_name)
        elif var_type == 'control_rate2':
            control_name = name[:-6]
            obj_path = 'control_rates:{0}_rate2'.format(control_name)
        elif var_type == 'design_parameter':
            obj_path = 'design_parameters:{0}'.format(name)
        elif var_type == 'input_parameter':
            obj_path = 'input_parameters:{0}_out'.format(name)
        else:
            # Failed to find variable, assume it is in the RHS
            obj_path = 'ode.{0}'.format(name)

        super(RungeKuttaPhase, self)._add_objective(obj_path, loc=loc, index=index, shape=shape,
                                                    ref=ref, ref0=ref0, adder=adder,
                                                    scaler=scaler,
                                                    parallel_deriv_color=parallel_deriv_color,
                                                    vectorize_derivs=vectorize_derivs)

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
