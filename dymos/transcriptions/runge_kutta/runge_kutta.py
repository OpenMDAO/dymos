from __future__ import division, print_function, absolute_import

import numpy as np

import openmdao.api as om
from six import iteritems

from ..transcription_base import TranscriptionBase
from .components import RungeKuttaStepsizeComp, RungeKuttaStateContinuityIterGroup, \
    RungeKuttaTimeseriesOutputComp, RungeKuttaPathConstraintComp, RungeKuttaControlContinuityComp
from ..common import TimeComp, EndpointConditionsComp
from ...utils.rk_methods import rk_methods
from ...utils.misc import CoerceDesvar, get_rate_units
from ...utils.constants import INF_BOUND
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData


class RungeKutta(TranscriptionBase):
    """
    The RungeKutta Transcription class.

    RungeKutta transcription in Dymos uses the RungeKutta-based shooting method which propagates
    the states from the phase initial time to the phase final time.
    """
    def initialize(self):

        self.options.declare('method', default='RK4', values=('RK4',),
                             desc='The integrator used within the explicit phase.')

        self.options.declare('k_solver_class', default=om.NonlinearBlockGS,
                             values=(om.NonlinearBlockGS, om.NewtonSolver, om.NonlinearRunOnce),
                             allow_none=True,
                             desc='The nonlinear solver class used to converge the numerical '
                                  'integration across each segment.')

        self.options.declare('k_solver_options', default={'iprint': -1}, types=(dict,),
                             desc='The options passed to the nonlinear solver used to converge the'
                                  'Runge-Kutta propagation across each step.')

    def setup_grid(self, phase):
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription='runge-kutta',
                                  transcription_order=self.options['method'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def setup_solvers(self, phase):
        """
        Add a NewtonSolver to converge continuity errors in the state between steps.

        Parameters
        ----------
        phase
            The phase to which this transcription instance applies.

        Returns
        -------

        """
        phase.nonlinear_solver = om.NewtonSolver()
        phase.nonlinear_solver.options['iprint'] = -1
        phase.nonlinear_solver.options['solve_subsystems'] = True
        phase.nonlinear_solver.options['err_on_maxiter'] = True
        phase.nonlinear_solver.linesearch = om.BoundsEnforceLS()

    def setup_time(self, phase):
        time_units = phase.time_options['units']
        num_seg = self.options['num_segments']
        rk_data = rk_methods[self.options['method']]
        num_nodes = num_seg * rk_data['num_stages']
        grid_data = self.grid_data

        super(RungeKutta, self).setup_time(phase)

        time_comp = TimeComp(num_nodes=num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau, units=time_units)

        phase.add_subsystem('time', time_comp, promotes_outputs=['*'], promotes_inputs=['*'])

        h_comp = RungeKuttaStepsizeComp(num_segments=num_seg,
                                        seg_rel_lengths=np.diff(grid_data.segment_ends),
                                        time_units=time_units)

        phase.add_subsystem('stepsize_comp',
                            subsys=h_comp,
                            promotes_inputs=['t_duration'],
                            promotes_outputs=['h'])

        if phase.time_options['targets']:
            time_tgts = phase.time_options['targets']

            phase.connect('time', ['rk_solve_group.ode.{0}'.format(t) for t in time_tgts],
                          src_indices=grid_data.subset_node_indices['all'])

            phase.connect('time', ['ode.{0}'.format(t) for t in time_tgts],
                          src_indices=grid_data.subset_node_indices['segment_ends'])

        if phase.time_options['time_phase_targets']:
            time_phase_tgts = phase.time_options['time_phase_targets']
            phase.connect('time_phase',
                          ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            phase.connect('time_phase',
                          ['ode.{0}'.format(t) for t in time_phase_tgts],
                          src_indices=grid_data.subset_node_indices['segment_ends'])

        if phase.time_options['t_initial_targets']:
            time_phase_tgts = phase.time_options['t_initial_targets']
            phase.connect('t_initial', ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            phase.connect('t_initial', ['ode.{0}'.format(t) for t in time_phase_tgts])

        if phase.time_options['t_duration_targets']:
            time_phase_tgts = phase.time_options['t_duration_targets']
            phase.connect('t_duration',
                          ['rk_solve_group.ode.{0}'.format(t) for t in time_phase_tgts])
            phase.connect('t_duration',
                          ['ode.{0}'.format(t) for t in time_phase_tgts])

    def setup_ode(self, phase):

        num_connected = len([s for s in phase.state_options
                             if phase.state_options[s]['connected_initial']])
        promoted_inputs = ['h'] if num_connected == 0 else ['h', 'initial_states:*']

        phase.add_subsystem('rk_solve_group',
                            RungeKuttaStateContinuityIterGroup(
                                num_segments=self.options['num_segments'],
                                method=self.options['method'],
                                state_options=phase.state_options,
                                time_units=phase.time_options['units'],
                                ode_class=phase.options['ode_class'],
                                ode_init_kwargs=phase.options['ode_init_kwargs'],
                                k_solver_class=self.options['k_solver_class'],
                                k_solver_options=self.options['k_solver_options']),
                            promotes_inputs=promoted_inputs,
                            promotes_outputs=['states:*', 'state_predict_comp.predicted_states:*'])

        # Since the RK Solve group evaluates the ODE at *predicted* state values, we need
        # to instantiate a second ODE group that will call the ODE at the actual integrated
        # state values so that we can accurate evaluate path and boundary constraints and
        # obtain timeseries for ODE outputs.
        phase.add_subsystem('ode',
                            phase.options['ode_class'](num_nodes=2*self.options['num_segments'],
                                                       **phase.options['ode_init_kwargs']))

    def _get_rate_source_path(self, state_name, phase, nodes=None, **kwargs):
        try:
            var = phase.state_options[state_name]['rate_source']
        except RuntimeError:
            raise ValueError('state \'{0}\' in phase \'{1}\' was not given a '
                             'rate_source'.format(state_name, phase.name))
        shape = phase.state_options[state_name]['shape']
        state_size = np.prod(shape)
        var_type = phase.classify_var(var)
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
            rate_path = 'state_predict_comp.predicted_states:{0}'.format(var)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'indep_control':
            rate_path = 'control_values:{0}'.format(var)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'input_control':
            rate_path = 'control_values:{0}'.format(var)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'control_rate':
            rate_path = 'control_rates:{0}'.format(var)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'control_rate2':
            rate_path = 'control_rates:{0}'.format(var)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'indep_polynomial_control':
            rate_path = 'polynomial_control_values:{0}'.format(var)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'input_polynomial_control':
            rate_path = 'polynomial_control_values:{0}'.format(var)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            rate_path = 'polynomial_control_rates:{0}_rate'.format(control_name)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            rate_path = 'polynomial_control_rates:{0}_rate2'.format(control_name)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))
        elif var_type == 'design_parameter':
            rate_path = 'design_parameters:{0}'.format(var)
            size = np.prod(phase.design_parameter_options[var]['shape'])
            src_idxs = np.zeros(num_segments * num_stages * size, dtype=int).reshape((num_segments,
                                                                                      num_stages,
                                                                                      state_size))
        elif var_type == 'input_parameter':
            rate_path = 'input_parameters:{0}_out'.format(var)
            size = np.prod(phase.input_parameter_options[var]['shape'])
            src_idxs = np.zeros(num_segments * num_stages * size, dtype=int).reshape((num_segments,
                                                                                      num_stages,
                                                                                      state_size))
        else:
            # Failed to find variable, assume it is in the ODE
            rate_path = 'rk_solve_group.ode.{0}'.format(var)
            state_size = np.prod(shape)
            size = num_segments * num_stages * state_size
            src_idxs = np.arange(size, dtype=int).reshape((num_segments, num_stages, state_size))

        return rate_path, src_idxs

    def setup_states(self, phase):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        num_seg = self.options['num_segments']
        num_state_input_nodes = num_seg + 1

        for state_name, options in iteritems(phase.state_options):
            size = np.prod(options['shape'])

            # Connect the states at the segment ends to the final ODE instance.
            row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
            row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
            src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
            if options['targets']:
                phase.connect('states:{0}'.format(state_name),
                              ['ode.{0}'.format(tgt) for tgt in options['targets']],
                              src_indices=src_idxs.ravel(), flat_src_indices=True)

            # Connect the state rate source to the k comp
            rate_path, src_idxs = self._get_rate_source_path(state_name, phase)

            phase.connect(rate_path,
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
                                         'phase {1}.'.format(state_name, phase.name))
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

                    phase.add_design_var(name='states:{0}'.format(state_name),
                                         lower=lb,
                                         upper=ub,
                                         scaler=coerce_desvar_option('scaler'),
                                         adder=coerce_desvar_option('adder'),
                                         ref0=coerce_desvar_option('ref0'),
                                         ref=coerce_desvar_option('ref'),
                                         indices=desvar_indices)

    def setup_controls(self, phase):
        super(RungeKutta, self).setup_controls(phase)
        grid_data = self.grid_data

        for name, options in iteritems(phase.control_options):
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            all_idxs = grid_data.subset_node_indices['all']
            segend_src_idxs = get_src_indices_by_row(segment_end_idxs, shape=options['shape'])
            all_src_idxs = get_src_indices_by_row(all_idxs, shape=options['shape'])

            if phase.control_options[name]['targets']:
                src_name = 'control_values:{0}'.format(name)
                targets = phase.control_options[name]['targets']
                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs.ravel(), flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs.ravel(), flat_src_indices=True)

            if phase.control_options[name]['rate_targets']:
                src_name = 'control_rates:{0}_rate'.format(name)
                targets = phase.control_options[name]['rate_targets']

                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

            if phase.control_options[name]['rate2_targets']:
                src_name = 'control_rates:{0}_rate2'.format(name)
                targets = phase.control_options[name]['rate2_targets']

                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

    def setup_polynomial_controls(self, phase):
        super(RungeKutta, self).setup_polynomial_controls(phase)
        grid_data = self.grid_data

        for name, options in iteritems(phase.polynomial_control_options):
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            all_idxs = grid_data.subset_node_indices['all']
            segend_src_idxs = get_src_indices_by_row(segment_end_idxs, shape=options['shape'])
            all_src_idxs = get_src_indices_by_row(all_idxs, shape=options['shape'])

            if phase.polynomial_control_options[name]['targets']:
                src_name = 'polynomial_control_values:{0}'.format(name)
                targets = phase.polynomial_control_options[name]['targets']
                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs.ravel(), flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs.ravel(), flat_src_indices=True)

            if phase.polynomial_control_options[name]['rate_targets']:
                src_name = 'polynomial_control_rates:{0}_rate'.format(name)
                targets = phase.polynomial_control_options[name]['rate_targets']

                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

            if phase.polynomial_control_options[name]['rate2_targets']:
                src_name = 'polynomial_control_rates:{0}_rate2'.format(name)
                targets = phase.polynomial_control_options[name]['rate2_targets']

                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

    def setup_defects(self, phase):
        """
        Setup the Continuity component as necessary.
        """
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        # Add the continuity constraint component if necessary
        if num_seg > 1 and phase.control_options:
            time_units = phase.time_options['units']

            phase.add_subsystem('continuity_comp',
                                RungeKuttaControlContinuityComp(grid_data=grid_data,
                                                                state_options=phase.state_options,
                                                                control_options=phase.control_options,
                                                                time_units=time_units),
                                promotes_inputs=['t_duration'])

            for name, options in iteritems(phase.control_options):
                # The sub-indices of control_disc indices that are segment ends
                segment_end_idxs = grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(segment_end_idxs, options['shape'], flat=True)

                phase.connect('control_values:{0}'.format(name),
                              'continuity_comp.controls:{0}'.format(name),
                              src_indices=src_idxs, flat_src_indices=True)

                phase.connect('control_rates:{0}_rate'.format(name),
                              'continuity_comp.control_rates:{}_rate'.format(name),
                              src_indices=src_idxs, flat_src_indices=True)

                phase.connect('control_rates:{0}_rate2'.format(name),
                              'continuity_comp.control_rates:{}_rate2'.format(name),
                              src_indices=src_idxs, flat_src_indices=True)

    def setup_endpoint_conditions(self, phase):

        jump_comp = phase.add_subsystem('indep_jumps', subsys=om.IndepVarComp(),
                                        promotes_outputs=['*'])

        jump_comp.add_output('initial_jump:time', val=0.0, units=phase.time_options['units'],
                             desc='discontinuity in time at the start of the phase')

        jump_comp.add_output('final_jump:time', val=0.0, units=phase.time_options['units'],
                             desc='discontinuity in time at the end of the phase')

        ic_comp = EndpointConditionsComp(loc='initial',
                                         time_options=phase.time_options,
                                         state_options=phase.state_options,
                                         control_options=phase.control_options)

        phase.add_subsystem(name='initial_conditions', subsys=ic_comp, promotes_outputs=['*'])

        fc_comp = EndpointConditionsComp(loc='final',
                                         time_options=phase.time_options,
                                         state_options=phase.state_options,
                                         control_options=phase.control_options)

        phase.add_subsystem(name='final_conditions', subsys=fc_comp, promotes_outputs=['*'])

        phase.connect('time', 'initial_conditions.initial_value:time')
        phase.connect('time', 'final_conditions.final_value:time')
        phase.connect('initial_jump:time', 'initial_conditions.initial_jump:time')
        phase.connect('final_jump:time', 'final_conditions.final_jump:time')

        for state_name, options in iteritems(phase.state_options):
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

            phase.connect('states:{0}'.format(state_name),
                          'initial_conditions.initial_value:{0}'.format(state_name))

            phase.connect('states:{0}'.format(state_name),
                          'final_conditions.final_value:{0}'.format(state_name))

            phase.connect('initial_jump:{0}'.format(state_name),
                          'initial_conditions.initial_jump:{0}'.format(state_name),
                          src_indices=ar, flat_src_indices=True)

            phase.connect('final_jump:{0}'.format(state_name),
                          'final_conditions.final_jump:{0}'.format(state_name),
                          src_indices=ar, flat_src_indices=True)

        for control_name, options in iteritems(phase.control_options):
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

            phase.connect('control_values:{0}'.format(control_name),
                          'initial_conditions.initial_value:{0}'.format(control_name))

            phase.connect('control_values:{0}'.format(control_name),
                          'final_conditions.final_value:{0}'.format(control_name))

            phase.connect('initial_jump:{0}'.format(control_name),
                          'initial_conditions.initial_jump:{0}'.format(control_name),
                          src_indices=ar, flat_src_indices=True)

            phase.connect('final_jump:{0}'.format(control_name),
                          'final_conditions.final_jump:{0}'.format(control_name),
                          src_indices=ar, flat_src_indices=True)

    def setup_path_constraints(self, phase):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        path_comp = None
        gd = self.grid_data
        time_units = phase.time_options['units']
        num_seg = gd.num_segments

        if phase._path_constraints:
            path_comp = RungeKuttaPathConstraintComp(grid_data=gd)
            phase.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(phase._path_constraints):
            con_units = options.get('units', None)
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                phase.connect(src_name='time',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name),
                              src_indices=self.grid_data.subset_node_indices['segment_ends'])
            elif var_type == 'time_phase':
                options['shape'] = (1,)
                options['units'] = time_units if con_units is None else con_units
                options['linear'] = True
                phase.connect(src_name='time_phase',
                              tgt_name='path_constraints.all_values:{0}'.format(con_name),
                              src_indices=self.grid_data.subset_node_indices['segment_ends'])
            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                state_units = phase.state_options[var]['units']
                options['shape'] = state_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = False
                row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
                row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
                src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
                phase.connect('states:{0}'.format(var),
                              'path_constraints.all_values:{0}'.format(var),
                              src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('indep_control', 'input_control'):
                control_shape = phase.control_options[var]['shape']
                control_units = phase.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True if var_type == 'indep_control' else False

                src_rows = self.grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'control_values:{0}'.format(var)

                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                phase.connect(src_name=src, tgt_name=tgt,
                              src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
                control_shape = phase.polynomial_control_options[var]['shape']
                control_units = phase.polynomial_control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = False

                src_rows = self.grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'polynomial_control_values:{0}'.format(var)

                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                phase.connect(src_name=src, tgt_name=tgt,
                              src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('control_rate', 'control_rate2'):
                if var.endswith('_rate'):
                    control_name = var[:-5]
                elif var.endswith('_rate2'):
                    control_name = var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                control_units = phase.control_options[control_name]['units']
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

                phase.connect(src_name=src, tgt_name=tgt,
                              src_indices=src_idxs, flat_src_indices=True)

            elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
                if var.endswith('_rate'):
                    control_name = var[:-5]
                elif var.endswith('_rate2'):
                    control_name = var[:-6]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                control_units = phase.polynomial_control_options[control_name]['units']
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

                phase.connect(src_name=src, tgt_name=tgt,
                              src_indices=src_idxs, flat_src_indices=True)

            else:
                # Failed to find variable, assume it is in the ODE
                options['linear'] = False

                if options['shape'] is None:
                    options['shape'] = (1,)

                src_rows = np.arange(num_seg * 2, dtype=int)
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])

                src = 'ode.{0}'.format(var)
                tgt = 'path_constraints.all_values:{0}'.format(con_name)

                phase.connect(src_name=src, tgt_name=tgt,
                              src_indices=src_idxs, flat_src_indices=True)

            kwargs = options.copy()
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def setup_timeseries_outputs(self, phase):

        gd = self.grid_data
        num_seg = gd.num_segments
        time_units = phase.time_options['units']
        timeseries_comp = RungeKuttaTimeseriesOutputComp(grid_data=gd)
        phase.add_subsystem('timeseries', subsys=timeseries_comp)
        src_idxs = get_src_indices_by_row(gd.subset_node_indices['segment_ends'], (1,))

        timeseries_comp._add_timeseries_output('time',
                                               var_class=phase.classify_var('time'),
                                               units=time_units)
        phase.connect(src_name='time', tgt_name='timeseries.segend_values:time',
                      src_indices=src_idxs, flat_src_indices=True)

        timeseries_comp._add_timeseries_output('time_phase',
                                               var_class=phase.classify_var('time_phase'),
                                               units=time_units)
        phase.connect(src_name='time_phase', tgt_name='timeseries.segend_values:time_phase',
                      src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.state_options):
            timeseries_comp._add_timeseries_output('states:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=options['units'])
            row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
            row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
            src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
            phase.connect(src_name='states:{0}'.format(name),
                          tgt_name='timeseries.segend_values:states:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('controls:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['segment_ends']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            phase.connect(src_name='control_values:{0}'.format(name),
                          tgt_name='timeseries.segend_values:controls:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

            # # Control rates
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            phase.connect(src_name='control_rates:{0}_rate'.format(name),
                          tgt_name='timeseries.segend_values:control_rates:{0}_rate'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

            # Control second derivatives
            timeseries_comp._add_timeseries_output('control_rates:{0}_rate2'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            phase.connect(src_name='control_rates:{0}_rate2'.format(name),
                          tgt_name='timeseries.segend_values:control_rates:{0}_rate2'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.polynomial_control_options):
            control_units = options['units']
            timeseries_comp._add_timeseries_output('polynomial_controls:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=control_units)
            src_rows = gd.subset_node_indices['segment_ends']
            src_idxs = get_src_indices_by_row(src_rows, options['shape'])
            phase.connect(src_name='polynomial_control_values:{0}'.format(name),
                          tgt_name='timeseries.segend_values:polynomial_controls:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

            # # Control rates
            timeseries_comp._add_timeseries_output('polynomial_control_rates:{0}_rate'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=1))
            phase.connect(src_name='polynomial_control_rates:{0}_rate'.format(name),
                          tgt_name='timeseries.segend_values:polynomial_control_rates'
                                   ':{0}_rate'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

            # Control second derivatives
            timeseries_comp._add_timeseries_output('polynomial_control_rates:'
                                                   '{0}_rate2'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=get_rate_units(control_units,
                                                                        time_units,
                                                                        deriv=2))
            phase.connect(src_name='polynomial_control_rates:{0}_rate2'.format(name),
                          tgt_name='timeseries.segend_values:polynomial_control_rates'
                                   ':{0}_rate2'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.design_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('design_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   shape=options['shape'],
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='design_parameters:{0}'.format(name),
                          tgt_name='timeseries.segend_values:design_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.input_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('input_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='input_parameters:{0}_out'.format(name),
                          tgt_name='timeseries.segend_values:input_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for name, options in iteritems(phase.traj_parameter_options):
            units = options['units']
            timeseries_comp._add_timeseries_output('traj_parameters:{0}'.format(name),
                                                   var_class=phase.classify_var(name),
                                                   units=units)

            src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
            src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

            phase.connect(src_name='traj_parameters:{0}_out'.format(name),
                          tgt_name='timeseries.segend_values:traj_parameters:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

        for var, options in iteritems(phase._timeseries_outputs):
            output_name = options['output_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            # Ignore any variables that we've already added (states, times, controls, etc)
            if var_type != 'ode':
                continue

            # Assume scalar shape if None, but check config will warn that it's inferred.
            if options['shape'] is None:
                options['shape'] = (1,)

            # Failed to find variable, assume it is in the RHS
            phase.connect(src_name='ode.{0}'.format(var),
                          tgt_name='timeseries.segend_values:{0}'.format(output_name))

            kwargs = options.copy()
            kwargs.pop('output_name', None)
            timeseries_comp._add_timeseries_output(output_name, var_type, **kwargs)

    def get_parameter_connections(self, name, phase):
        """
        Returns a list containing tuples of each path and related indices to which the
        given design variable name is to be connected.

        Parameters
        ----------
        name : str
            The name of the parameter for which connection info is desired.
        phase : Phase
            The Phase instance with which this transcription is associated.

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

        parameter_options = phase.design_parameter_options.copy()
        parameter_options.update(phase.input_parameter_options)
        parameter_options.update(phase.traj_parameter_options)
        parameter_options.update(phase.control_options)

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

    def _get_boundary_constraint_src(self, var, loc, phase):
        """
        Get the source for boundary constraint values.

        Parameters
        ----------
        var : str
            The name of the variable whose source information is desired.
        loc : str
            The location of the boundary constraint ('initial' or 'final')
        phase : Phase
            The Phase instance with which this transcription is associated.

        Returns
        -------
        constraint_path, shape, units, linear

        """
        # Determine the path to the variable which we will be constraining
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

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
            state_shape = phase.state_options[var]['shape']
            state_units = phase.state_options[var]['units']
            shape = state_shape
            units = state_units
            linear = True if loc == 'initial' and phase.state_options[var]['fix_initial'] or \
                loc == 'final' and phase.state_options[var]['fix_final'] else False
            constraint_path = 'states:{0}'.format(var)
        elif var_type in ('indep_control', 'input_control'):
            control_shape = phase.control_options[var]['shape']
            control_units = phase.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'control_values:{0}'.format(var)
        elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
            control_shape = phase.polynomial_control_options[var]['shape']
            control_units = phase.polynomial_control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'polynomial_control_values:{0}'.format(var)
        elif var_type == 'design_parameter':
            control_shape = phase.design_parameter_options[var]['shape']
            control_units = phase.design_parameter_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'design_parameters:{0}'.format(var)
        elif var_type == 'input_parameter':
            control_shape = phase.input_parameter_options[var]['shape']
            control_units = phase.input_parameter_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'input_parameters:{0}_out'.format(var)
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5] if var_type == 'control_rate' else var[:-6]
            control_shape = phase.control_options[control_var]['shape']
            control_units = phase.control_options[control_var]['units']
            d = 1 if var_type == 'control_rate' else 2
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5] if var_type == 'polynomial_control_rate' else var[:-6]
            control_shape = phase.polynomial_control_options[control_var]['shape']
            control_units = phase.polynomial_control_options[control_var]['units']
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
