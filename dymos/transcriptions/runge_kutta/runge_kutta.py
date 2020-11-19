import numpy as np

import openmdao.api as om
from openmdao.utils.general_utils import warn_deprecation

from ..transcription_base import TranscriptionBase
from .components import RungeKuttaStepsizeComp, RungeKuttaStateContinuityIterGroup, \
    RungeKuttaTimeseriesOutputComp, RungeKuttaControlContinuityComp
from ..common import TimeComp, PathConstraintComp
from ...utils.rk_methods import rk_methods
from ...utils.misc import CoerceDesvar, get_rate_units, get_target_metadata,\
    get_source_metadata, _unspecified
from ...utils.introspection import get_targets
from ...utils.constants import INF_BOUND
from ...utils.indexing import get_src_indices_by_row
from ..grid_data import GridData
from fnmatch import filter


class RungeKutta(TranscriptionBase):
    """
    The RungeKutta Transcription class.

    RungeKutta transcription in Dymos uses the RungeKutta-based shooting method which propagates
    the states from the phase initial time to the phase final time.
    """
    def __init__(self, **kwargs):
        super(RungeKutta, self).__init__(**kwargs)
        self._rhs_source = 'ode'

        msg = 'The RungeKutta transcription is deprecated and will be removed in Dymos v1.0.0.\n' \
              'For equivalent behavior, users should switch to ' \
              'GaussLobatto(order=3, solve_segments=True)'
        warn_deprecation(msg)

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

    def init_grid(self):
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
        phase.nonlinear_solver.options['err_on_non_converge'] = True
        phase.nonlinear_solver.linesearch = om.BoundsEnforceLS()

    def configure_solvers(self, phase):
        pass

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

    def configure_time(self, phase):
        super(RungeKutta, self).configure_time(phase)

        phase.time.configure_io()
        phase.stepsize_comp.configure_io()

        options = phase.time_options

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, usr_tgts, dynamic in [('time', options['targets'], True),
                                        ('time_phase', options['time_phase_targets'], True),
                                        ('t_initial', options['t_initial_targets'], False),
                                        ('t_duration', options['t_duration_targets'], False)]:

            targets = get_targets(phase.ode, name=name, user_targets=usr_tgts)
            if targets:
                all_src_idxs = self.grid_data.subset_node_indices['all'] if dynamic else None
                end_src_idxs = self.grid_data.subset_node_indices['segment_ends'] if dynamic else None
                phase.connect(name, ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs)
                phase.connect(name, ['ode.{0}'.format(t) for t in targets],
                              src_indices=end_src_idxs)

    def setup_ode(self, phase):
        phase.add_subsystem('rk_solve_group',
                            RungeKuttaStateContinuityIterGroup(
                                num_segments=self.options['num_segments'],
                                method=self.options['method'],
                                state_options=phase.state_options,
                                time_units=phase.time_options['units'],
                                ode_class=phase.options['ode_class'],
                                ode_init_kwargs=phase.options['ode_init_kwargs'],
                                k_solver_class=self.options['k_solver_class'],
                                k_solver_options=self.options['k_solver_options']))

        # Since the RK Solve group evaluates the ODE at *predicted* state values, we need
        # to instantiate a second ODE group that will call the ODE at the actual integrated
        # state values so that we can accurate evaluate path and boundary constraints and
        # obtain timeseries for ODE outputs.
        phase.add_subsystem('ode',
                            phase.options['ode_class'](num_nodes=2*self.options['num_segments'],
                                                       **phase.options['ode_init_kwargs']))

    def configure_ode(self, phase):
        phase.rk_solve_group.configure_io()

        num_connected = len([s for s in phase.state_options
                             if phase.state_options[s]['connected_initial']])

        phase.promotes('rk_solve_group',
                       inputs=['h'] if num_connected == 0 else ['h', 'initial_states:*'],
                       outputs=['states:*', 'state_predict_comp.predicted_states:*'])

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
        elif var_type == 'parameter':
            rate_path = 'parameters:{0}'.format(var)
            size = np.prod(phase.parameter_options[var]['shape'])
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
        pass

    def configure_states(self, phase):
        num_seg = self.options['num_segments']
        num_state_input_nodes = num_seg + 1

        for state_name, options in phase.state_options.items():

            self._configure_state_introspection(state_name, options, phase)

            size = np.prod(options['shape'])

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

        num_seg = self.options['num_segments']

        for state_name, options in phase.state_options.items():
            shape = options['shape']

            # Connect the states at the segment ends to the final ODE instance.
            row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
            row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
            src_idxs = get_src_indices_by_row(row_idxs, shape)

            if shape == (1,):
                src_idxs = src_idxs.ravel()

            targets = get_targets(ode=phase.ode, name=state_name, user_targets=options['targets'])

            if targets:
                phase.connect('states:{0}'.format(state_name),
                              ['ode.{0}'.format(tgt) for tgt in targets],
                              src_indices=src_idxs, flat_src_indices=True)

            # Connect the state rate source to the k comp
            rate_path, src_idxs = self._get_rate_source_path(state_name, phase)

            phase.connect(rate_path,
                          'rk_solve_group.k_comp.f:{0}'.format(state_name),
                          src_indices=src_idxs,
                          flat_src_indices=True)

    def setup_controls(self, phase):
        super(RungeKutta, self).setup_controls(phase)

    def configure_controls(self, phase):
        super(RungeKutta, self).configure_controls(phase)

        grid_data = self.grid_data

        for name, options in phase.control_options.items():
            shape = options['shape']
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            all_idxs = grid_data.subset_node_indices['all']
            segend_src_idxs = get_src_indices_by_row(segment_end_idxs, shape=shape)
            all_src_idxs = get_src_indices_by_row(all_idxs, shape=shape)

            if shape == (1,):
                segend_src_idxs = segend_src_idxs.ravel()
                all_src_idxs = all_src_idxs.ravel()

            targets = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])
            if targets:
                src_name = 'control_values:{0}'.format(name)
                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)
                phase.connect(src_name,
                              ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

            targets = get_targets(ode=phase.ode, name=f'{name}_rate',
                                  user_targets=options['rate_targets'])
            if targets:
                src_name = 'control_rates:{0}_rate'.format(name)

                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

            targets = get_targets(ode=phase.ode, name=f'{name}_rate2',
                                  user_targets=options['rate2_targets'])
            if targets:
                src_name = 'control_rates:{0}_rate2'.format(name)

                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

    def setup_polynomial_controls(self, phase):
        super(RungeKutta, self).setup_polynomial_controls(phase)

    def configure_polynomial_controls(self, phase):
        super(RungeKutta, self).configure_polynomial_controls(phase)
        grid_data = self.grid_data

        for name, options in phase.polynomial_control_options.items():
            shape = options['shape']
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            all_idxs = grid_data.subset_node_indices['all']
            segend_src_idxs = get_src_indices_by_row(segment_end_idxs, shape=shape)
            all_src_idxs = get_src_indices_by_row(all_idxs, shape=shape)

            if shape == (1,):
                segend_src_idxs = segend_src_idxs.ravel()
                all_src_idxs = all_src_idxs.ravel()

            targets = get_targets(phase.ode, name=name, user_targets=options['targets'])
            if targets:
                src_name = 'polynomial_control_values:{0}'.format(name)
                phase.connect(src_name,
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)

                phase.connect(src_name,
                              ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

            targets = get_targets(ode=phase.ode, name=f'{name}_rate',
                                  user_targets=options['rate_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate',
                              ['ode.{0}'.format(t) for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)
                phase.connect(f'polynomial_control_rates:{name}_rate',
                              ['rk_solve_group.ode.{0}'.format(t) for t in targets],
                              src_indices=all_src_idxs, flat_src_indices=True)

            targets = get_targets(ode=phase.ode, name=f'{name}_rate2',
                                  user_targets=options['rate2_targets'])
            if targets:
                phase.connect(f'polynomial_control_rates:{name}_rate2',
                              [f'ode.{t}' for t in targets],
                              src_indices=segend_src_idxs, flat_src_indices=True)
                phase.connect(f'polynomial_control_rates:{name}_rate2',
                              [f'rk_solve_group.ode.{t}' for t in targets],
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

    def configure_defects(self, phase):
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        # Add the continuity constraint component if necessary
        if num_seg > 1 and phase.control_options:
            phase.continuity_comp.configure_io()

            for name, options in phase.control_options.items():
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

    def setup_path_constraints(self, phase):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.  This overrides the default transcription path constraints at
        all nodes and only applies them at the segment end points, the only points in an RK segment
        where all of the variables are valid.
        """
        gd = self.grid_data

        if phase._path_constraints:
            path_comp = PathConstraintComp(num_nodes=gd.subset_num_nodes['segment_ends'])
            phase.add_subsystem('path_constraints', subsys=path_comp)

    def configure_path_constraints(self, phase):
        super(RungeKutta, self).configure_path_constraints(phase)

        gd = self.grid_data
        num_seg = gd.num_segments
        num_seg_ends = self.grid_data.subset_num_nodes['segment_ends']
        seg_end_idxs = self.grid_data.subset_node_indices['segment_ends']

        for var, options in phase._path_constraints.items():
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = phase.classify_var(var)

            if var_type == 'time':
                src = 'time'
                tgt = f'path_constraints.all_values:{con_name}'
                src_idxs = np.reshape(seg_end_idxs, newshape=(num_seg_ends, 1))
                flat_src_idxs = False

            elif var_type == 'time_phase':
                src = 'time_phase'
                tgt = f'path_constraints.all_values:{con_name}'
                src_idxs = np.reshape(seg_end_idxs, newshape=(num_seg_ends, 1))
                flat_src_idxs = False

            elif var_type == 'state':
                state_shape = phase.state_options[var]['shape']
                row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
                row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
                src_idxs = get_src_indices_by_row(row_idxs, state_shape)
                flat_src_idxs = True
                src = f'states:{var}'
                tgt = f'path_constraints.all_values:{var}'

            elif var_type in ('indep_control', 'input_control'):
                control_shape = phase.control_options[var]['shape']

                src_rows = self.grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, shape=control_shape)
                flat_src_idxs = True

                src = f'control_values:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type in ('indep_polynomial_control', 'input_polynomial_control'):
                shape = phase.polynomial_control_options[var]['shape']
                src_idxs = get_src_indices_by_row(seg_end_idxs, shape=shape)
                flat_src_idxs = True

                src = f'polynomial_control_values:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type in ('control_rate', 'control_rate2'):
                control_name = var[:-5] if var.endswith('_rate') else var[:-6]
                control_shape = phase.control_options[control_name]['shape']
                src_idxs = get_src_indices_by_row(seg_end_idxs, shape=control_shape)
                flat_src_idxs = True

                src = f'control_rates:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
                control_name = var[:-5] if var.endswith('_rate') else var[:-6]
                control_shape = phase.polynomial_control_options[control_name]['shape']
                src_idxs = get_src_indices_by_row(seg_end_idxs, shape=control_shape)
                flat_src_idxs = True

                src = f'polynomial_control_rates:{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            else:
                src_rows = np.arange(num_seg * 2, dtype=int)
                src_idxs = get_src_indices_by_row(src_rows, shape=options['shape'])
                flat_src_idxs = True

                src = f'ode.{var}'
                tgt = f'path_constraints.all_values:{con_name}'

            phase.connect(src_name=src, tgt_name=tgt,
                          src_indices=src_idxs, flat_src_indices=flat_src_idxs)

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data

        for name, options in phase._timeseries.items():

            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = RungeKuttaTimeseriesOutputComp(input_grid_data=gd,
                                                             output_grid_data=ogd,
                                                             output_subset=options['subset'])
            phase.add_subsystem(name, subsys=timeseries_comp)

    def configure_timeseries_outputs(self, phase):
        gd = self.grid_data
        num_seg = gd.num_segments
        time_units = phase.time_options['units']

        for timeseries_name, timeseries_options in phase._timeseries.items():
            timeseries_comp = phase._get_subsystem(timeseries_name)

            src_idxs = get_src_indices_by_row(gd.subset_node_indices['segment_ends'], (1,))

            timeseries_comp._add_output_configure('time',
                                                  shape=(1,),
                                                  units=time_units,
                                                  desc='time')

            timeseries_comp._add_output_configure('time_phase',
                                                  shape=(1,),
                                                  units=time_units,
                                                  desc='phase elapsed time')

            phase.connect(src_name='time', tgt_name=f'{timeseries_name}.input_values:time',
                          src_indices=src_idxs, flat_src_indices=True)

            phase.connect(src_name='time_phase', tgt_name=f'{timeseries_name}.input_values:time_phase',
                          src_indices=src_idxs, flat_src_indices=True)

            for state_name, options in phase.state_options.items():
                row_idxs = np.repeat(np.arange(1, num_seg, dtype=int), repeats=2)
                row_idxs = np.concatenate(([0], row_idxs, [num_seg]))
                src_idxs = get_src_indices_by_row(row_idxs, options['shape'])

                timeseries_comp._add_output_configure(f'states:{state_name}',
                                                      shape=options['shape'],
                                                      units=options['units'],
                                                      desc=options['desc'])

                phase.connect(src_name=f'states:{state_name}',
                              tgt_name=f'{timeseries_name}.input_values:states:{state_name}',
                              src_indices=src_idxs, flat_src_indices=True)

            for control_name, options in phase.control_options.items():
                control_units = options['units']
                src_rows = gd.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])

                timeseries_comp._add_output_configure(f'controls:{control_name}',
                                                      shape=options['shape'],
                                                      units=control_units,
                                                      desc=options['desc'])

                phase.connect(src_name='control_values:{0}'.format(control_name),
                              tgt_name=f'{timeseries_name}.input_values:controls:{control_name}',
                              src_indices=src_idxs, flat_src_indices=True)

                # Control rates
                timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units,
                                                                           deriv=1),
                                                      desc=f'first time-derivative of control {control_name}')

                phase.connect(src_name=f'control_rates:{control_name}_rate',
                              tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate',
                              src_indices=src_idxs, flat_src_indices=True)

                # Control second derivatives
                timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate2',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units,
                                                                           deriv=2),
                                                      desc=f'second time-derivative of control {control_name}')

                phase.connect(src_name=f'control_rates:{control_name}_rate2',
                              tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate2',
                              src_indices=src_idxs, flat_src_indices=True)

            for control_name, options in phase.polynomial_control_options.items():
                src_rows = gd.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(src_rows, options['shape'])

                control_units = options['units']

                timeseries_comp._add_output_configure(f'polynomial_controls:{control_name}',
                                                      shape=options['shape'],
                                                      units=control_units,
                                                      desc=options['desc'])

                phase.connect(src_name=f'polynomial_control_values:{control_name}',
                              tgt_name=f'{timeseries_name}.input_values:polynomial_controls:{control_name}',
                              src_indices=src_idxs, flat_src_indices=True)

                # Control rates
                timeseries_comp._add_output_configure('polynomial_control_rates:{0}_rate'.format(control_name),
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units, time_units,
                                                                           deriv=1),
                                                      desc=f'first time-derivative of polynomial '
                                                           f'control {control_name}')

                # Control rates
                phase.connect(src_name=f'polynomial_control_rates:{control_name}_rate',
                              tgt_name=f'{timeseries_name}.input_values:polynomial_control_rates:{control_name}_rate',
                              src_indices=src_idxs, flat_src_indices=True)

                # Control second derivatives
                timeseries_comp._add_output_configure(f'polynomial_control_rates:{control_name}_rate2',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units, time_units,
                                                                           deriv=2),
                                                      desc=f'second time-derivative of polynomial '
                                                           f'control {control_name}')

                phase.connect(src_name=f'polynomial_control_rates:{control_name}_rate2',
                              tgt_name=f'{timeseries_name}.input_values:polynomial_control_rates:{control_name}_rate2',
                              src_indices=src_idxs, flat_src_indices=True)

            for param_name, options in phase.parameter_options.items():
                if options['include_timeseries']:
                    src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
                    src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])

                    prom_name = 'parameters:{0}'.format(param_name)
                    tgt_name = 'input_values:parameters:{0}'.format(param_name)

                    timeseries_comp._add_output_configure(f'parameters:{param_name}',
                                                          shape=options['shape'],
                                                          units=options['units'],
                                                          desc='')

                    phase.promotes(timeseries_name, inputs=[(tgt_name, prom_name)],
                                   src_indices=src_idxs, flat_src_indices=True)

            for var, options in timeseries_options['outputs'].items():
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

                    try:
                        shape, units = get_source_metadata(phase.ode, src=v,
                                                           user_units=units,
                                                           user_shape=options['shape'])
                    except ValueError:
                        raise ValueError(f'Timeseries output {v} is not a known variable in'
                                         f' the phase {phase.pathname} nor is it a known output of '
                                         f' the ODE.')

                    try:
                        timeseries_comp._add_output_configure(output_name, units, shape, desc='')
                    except ValueError as e:  # OK if it already exists
                        if 'already exists' in str(e):
                            continue
                        else:
                            raise e

                    # Failed to find variable, assume it is in the RHS
                    phase.connect(src_name=f'ode.{v}',
                                  tgt_name=f'{timeseries_name}.input_values:{output_name}')

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

        if name in phase.parameter_options:
            options = phase.parameter_options[name]
            ode_tgts = get_targets(ode=phase.ode, name=name, user_targets=options['targets'])
            dynamic = options['dynamic']
            shape = options['shape']

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
        elif var_type == 'parameter':
            control_shape = phase.parameter_options[var]['shape']
            control_units = phase.parameter_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'parameters:{0}'.format(var)
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
