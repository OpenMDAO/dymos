from __future__ import division, print_function, absolute_import

from collections.abc import Iterable

import numpy as np
from dymos.transcriptions.common import EndpointConditionsComp

import openmdao.api as om
from six import iteritems

from ..transcription_base import TranscriptionBase
from ..common import TimeComp
from .components import StateIndependentsComp, StateInterpComp, CollocationComp
from ...utils.misc import CoerceDesvar, get_rate_units
from ...utils.constants import INF_BOUND
from ...utils.indexing import get_src_indices_by_row


class PseudospectralBase(TranscriptionBase):
    """
    Base class for the pseudospectral transcriptions.
    """
    def initialize(self):
        self.options.declare(name='solve_segments', default=False, types=bool,
                             desc='default value for solve_segments for all states in the phase')

    def setup_time(self, phase):
        time_units = phase.time_options['units']
        grid_data = self.grid_data

        super(PseudospectralBase, self).setup_time(phase)

        time_comp = TimeComp(num_nodes=grid_data.num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau, units=time_units)

        phase.add_subsystem('time', time_comp, promotes_inputs=['*'], promotes_outputs=['*'])

    def setup_states(self, phase):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']

        self.any_solved_segs = False
        self.any_connected_opt_segs = False
        for name, options in iteritems(phase.state_options):
            if options['solve_segments'] is None:
                options['solve_segments'] = self.options['solve_segments']

            if options['solve_segments']:
                self.any_solved_segs = True
            elif options['connected_initial']:
                self.any_connected_opt_segs = True

        if self.any_solved_segs or self.any_connected_opt_segs:
            indep = StateIndependentsComp(grid_data=grid_data,
                                          state_options=phase.state_options)

            for name, options in iteritems(phase.state_options):
                if options['solve_segments']:
                    phase.connect('collocation_constraint.defects:{0}'.format(name),
                                  'indep_states.defects:{0}'.format(name))

        else:
            indep = om.IndepVarComp()

            for name, options in iteritems(phase.state_options):
                if not options['solve_segments'] and not options['connected_initial']:
                    indep.add_output(name='states:{0}'.format(name),
                                     shape=(num_state_input_nodes, np.prod(options['shape'])),
                                     units=options['units'])

        num_connected = len([s for (s, opts) in iteritems(phase.state_options) if opts['connected_initial']])
        prom_inputs = ['initial_states:*'] if num_connected > 0 else None
        phase.add_subsystem('indep_states', indep, promotes_inputs=prom_inputs,
                            promotes_outputs=['*'])

        # add all the des-vars (either from the IndepVarComp or from the indep-var-like
        # outputs of the collocation comp)
        for name, options in iteritems(phase.state_options):
            size = np.prod(options['shape'])
            if options['opt']:
                if options['solve_segments']:
                    desvar_indices = list(indep.state_idx_map[name]['indep'])
                else:
                    desvar_indices = list(range(size * num_state_input_nodes))

            if options['fix_initial']:
                if options['initial_bounds'] is not None:
                    raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                     'initial_bounds for state {0}'.format(name))
                if isinstance(options['fix_initial'], Iterable):
                    idxs_to_fix = np.where(np.asarray(options['fix_initial']))[0]
                    for idx_to_fix in reversed(sorted(idxs_to_fix)):
                        del desvar_indices[idx_to_fix]
                else:
                    del desvar_indices[:size]

            elif options['connected_initial'] and not options['solve_segments']:
                del desvar_indices[:size]

            if options['fix_final']:
                if options['final_bounds'] is not None:
                    raise ValueError('Cannot specify \'fix_final=True\' and specify '
                                     'final_bounds for state {0}'.format(name))
                if isinstance(options['fix_final'], Iterable):
                    idxs_to_fix = np.where(np.asarray(options['fix_final']))[0]
                    for idx_to_fix in reversed(sorted(idxs_to_fix)):
                        del desvar_indices[-size + idx_to_fix]
                else:
                    del desvar_indices[-size:]

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

                if options['final_bounds'] is not None:
                    lb[-1] = options['final_bounds'][0]
                    ub[-1] = options['final_bounds'][-1]

                phase.add_design_var(name='states:{0}'.format(name),
                                     lower=lb,
                                     upper=ub,
                                     scaler=coerce_desvar_option('scaler'),
                                     adder=coerce_desvar_option('adder'),
                                     ref0=coerce_desvar_option('ref0'),
                                     ref=coerce_desvar_option('ref'),
                                     indices=desvar_indices)

    def setup_ode(self, phase):
        grid_data = self.grid_data
        transcription = grid_data.transcription
        time_units = phase.time_options['units']
        map_input_indices_to_disc = grid_data.input_maps['state_input_to_disc']
        num_input_nodes = grid_data.subset_num_nodes['state_input']

        phase.add_subsystem('state_interp',
                            subsys=StateInterpComp(grid_data=grid_data,
                                                   state_options=phase.state_options,
                                                   time_units=time_units,
                                                   transcription=transcription))

        phase.connect('dt_dstau', 'state_interp.dt_dstau',
                      src_indices=grid_data.subset_node_indices['col'])

        for name, options in iteritems(phase.state_options):
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')

            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            phase.connect('states:{0}'.format(name),
                          'state_interp.state_disc:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

    def setup_defects(self, phase):
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        time_units = phase.time_options['units']

        phase.add_subsystem('collocation_constraint',
                            CollocationComp(grid_data=grid_data,
                                            state_options=phase.state_options,
                                            time_units=time_units))

        phase.connect('dt_dstau', ('collocation_constraint.dt_dstau'),
                      src_indices=grid_data.subset_node_indices['col'])

        # Add the continuity constraint component if necessary
        if num_seg > 1:
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            state_disc_idxs = grid_data.subset_node_indices['state_disc']

            if not self.options['compressed']:
                state_input_subidxs = np.where(np.in1d(state_disc_idxs, segment_end_idxs))[0]

                for name, options in iteritems(phase.state_options):
                    shape = options['shape']
                    flattened_src_idxs = get_src_indices_by_row(state_input_subidxs, shape=shape,
                                                                flat=True)
                    phase.connect('states:{0}'.format(name),
                                  'continuity_comp.states:{}'.format(name),
                                  src_indices=flattened_src_idxs, flat_src_indices=True)

            for name, options in iteritems(phase.control_options):
                control_src_name = 'control_values:{0}'.format(name)

                # The sub-indices of control_disc indices that are segment ends
                segment_end_idxs = grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(segment_end_idxs, options['shape'], flat=True)

                phase.connect(control_src_name,
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

        phase.connect('initial_jump:time',
                      'initial_conditions.initial_jump:time')

        phase.connect('final_jump:time',
                      'final_conditions.final_jump:time')

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

    def setup_solvers(self, phase):
        if self.any_solved_segs:
            newton = phase.nonlinear_solver = om.NewtonSolver()
            newton.options['solve_subsystems'] = True
            newton.options['maxiter'] = 100
            newton.options['iprint'] = -1
            newton.linesearch = om.BoundsEnforceLS()
            phase.linear_solver = om.DirectSolver()

    def _get_boundary_constraint_src(self, var, loc, phase):
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
            linear = True if loc == 'initial' and not phase.state_options[var]['connected_initial'] \
                or loc == 'final' and not phase.state_options[var]['solve_segments'] else False
            constraint_path = 'states:{0}'.format(var)
        elif var_type in 'indep_control':
            control_shape = phase.control_options[var]['shape']
            control_units = phase.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'control_values:{0}'.format(var)
        elif var_type == 'input_control':
            control_shape = phase.control_options[var]['shape']
            control_units = phase.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'control_values:{0}'.format(var)
        elif var_type in 'indep_polynomial_control':
            control_shape = phase.polynomial_control_options[var]['shape']
            control_units = phase.polynomial_control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'polynomial_control_values:{0}'.format(var)
        elif var_type == 'input_polynomial_control':
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
            d = 2 if var_type == 'control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5]
            control_shape = phase.polynomial_control_options[control_var]['shape']
            control_units = phase.polynomial_control_options[control_var]['units']
            d = 2 if var_type == 'polynomial_control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'polynomial_control_rates:{0}'.format(var)
        else:
            # Failed to find variable, assume it is in the RHS
            if self.grid_data.transcription == 'gauss-lobatto':
                constraint_path = 'rhs_disc.{0}'.format(var)
            elif self.grid_data.transcription == 'radau-ps':
                constraint_path = 'rhs_all.{0}'.format(var)
            else:
                raise ValueError('Invalid transcription')
            shape = None
            units = None
            linear = False

        return constraint_path, shape, units, linear
