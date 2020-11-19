from collections.abc import Iterable

import numpy as np

import openmdao.api as om
from ..transcription_base import TranscriptionBase
from ..common import TimeComp, PseudospectralTimeseriesOutputComp
from .components import StateIndependentsComp, StateInterpComp, CollocationComp
from ...utils.misc import CoerceDesvar, get_rate_units, _unspecified, \
    get_target_metadata, get_source_metadata
from ...utils.introspection import get_targets, get_state_target_metadata
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

    def configure_time(self, phase):
        super(PseudospectralBase, self).configure_time(phase)
        phase.time.configure_io()

    def setup_states(self, phase):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        grid_data = self.grid_data

        self.any_solved_segs = False
        self.any_connected_opt_segs = False
        for name, options in phase.state_options.items():
            # Transcription solve_segments overrides state solve_segments if its not set
            if options['solve_segments'] is None:
                options['solve_segments'] = self.options['solve_segments']

            if options['solve_segments']:
                self.any_solved_segs = True
            elif options['connected_initial']:
                self.any_connected_opt_segs = True

        if self.any_solved_segs or self.any_connected_opt_segs:
            indep = StateIndependentsComp(grid_data=grid_data,
                                          state_options=phase.state_options)
        else:
            indep = om.IndepVarComp()

        num_connected = len([s for (s, opts) in phase.state_options.items() if opts['connected_initial']])
        prom_inputs = ['initial_states:*'] if num_connected > 0 else None
        phase.add_subsystem('indep_states', indep, promotes_inputs=prom_inputs,
                            promotes_outputs=['*'])

    def configure_states(self, phase):
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']
        indep = phase.indep_states

        # add all the des-vars (either from the IndepVarComp or from the indep-var-like
        # outputs of the collocation comp)
        for name, options in phase.state_options.items():

            self._configure_state_introspection(name, options, phase)

            size = np.prod(options['shape'])
            # In certain cases, we put an output on the IVC.
            if isinstance(indep, om.IndepVarComp):
                if not options['solve_segments'] and not options['connected_initial']:
                    indep.add_output(name='states:{0}'.format(name),
                                     shape=(num_state_input_nodes, size),
                                     units=options['units'])

            if options['opt']:
                if options['solve_segments']:
                    # If we are using a solver on the defects, then our design variables
                    # are the first nodes in each segment.
                    # For instance, for two 5th order radau segments, the indep indices are [0, 6].
                    # If we have vectorized states of size n, then there are n design variables
                    # at each node.  For instance, with n=2, the desvar indices are [0, 1, 12, 13]
                    num_seg = grid_data.num_segments
                    # Get the desvar node indices
                    desvar_node_idxs = np.asarray(indep.state_idx_map[name]['indep'])
                    # In compressed transcription, the desvar indices are just the first
                    # index for each element in the state shape
                    if self.options['compressed']:
                        desvar_indices = np.arange(size, dtype=int)
                    else:
                        # In uncompressed transcription, we need desvar_indices repeated
                        # once for each segment, with the number of nodes in all but the
                        # last segments added to it
                        desvar_indices = size * np.repeat(desvar_node_idxs, size) + \
                            np.tile(np.arange(size, dtype=int), num_seg)
                    desvar_indices = list(desvar_indices)
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

        if not isinstance(indep, om.IndepVarComp):
            indep.configure_io()

        if self.any_solved_segs or self.any_connected_opt_segs:
            for name, options in phase.state_options.items():
                if options['solve_segments']:
                    phase.connect('collocation_constraint.defects:{0}'.format(name),
                                  'indep_states.defects:{0}'.format(name))

    def setup_ode(self, phase):
        grid_data = self.grid_data
        transcription = grid_data.transcription
        time_units = phase.time_options['units']

        phase.add_subsystem('state_interp',
                            subsys=StateInterpComp(grid_data=grid_data,
                                                   state_options=phase.state_options,
                                                   time_units=time_units,
                                                   transcription=transcription))

    def configure_ode(self, phase):
        grid_data = self.grid_data
        map_input_indices_to_disc = grid_data.input_maps['state_input_to_disc']

        phase.state_interp.configure_io()

        phase.connect('dt_dstau', 'state_interp.dt_dstau',
                      src_indices=grid_data.subset_node_indices['col'])

        for name, options in phase.state_options.items():
            size = np.prod(options['shape'])

            phase.connect('states:{0}'.format(name),
                          'state_interp.state_disc:{0}'.format(name),
                          src_indices=om.slicer[map_input_indices_to_disc, ...])

    def setup_defects(self, phase):
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data

        time_units = phase.time_options['units']

        phase.add_subsystem('collocation_constraint',
                            CollocationComp(grid_data=grid_data,
                                            state_options=phase.state_options,
                                            time_units=time_units))

    def configure_defects(self, phase):
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        phase.connect('dt_dstau', ('collocation_constraint.dt_dstau'),
                      src_indices=grid_data.subset_node_indices['col'])

        phase.collocation_constraint.configure_io()

        # Add the continuity constraint component if necessary
        if num_seg > 1:
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            state_disc_idxs = grid_data.subset_node_indices['state_disc']

            if not self.options['compressed']:
                state_input_subidxs = np.where(np.in1d(state_disc_idxs, segment_end_idxs))[0]

                for name, options in phase.state_options.items():
                    shape = options['shape']
                    flattened_src_idxs = get_src_indices_by_row(state_input_subidxs, shape=shape,
                                                                flat=True)
                    phase.connect('states:{0}'.format(name),
                                  'continuity_comp.states:{}'.format(name),
                                  src_indices=flattened_src_idxs, flat_src_indices=True)

            for name, options in phase.control_options.items():
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

    def setup_solvers(self, phase):
        if self.any_solved_segs:
            newton = phase.nonlinear_solver = om.NewtonSolver()
            newton.options['solve_subsystems'] = True
            newton.options['maxiter'] = 100
            newton.options['iprint'] = -1
            newton.linesearch = om.BoundsEnforceLS()
            phase.linear_solver = om.DirectSolver()

    def configure_solvers(self, phase):
        pass

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data

        for name, options in phase._timeseries.items():
            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = PseudospectralTimeseriesOutputComp(input_grid_data=gd,
                                                                 output_grid_data=ogd,
                                                                 output_subset=options['subset'])
            phase.add_subsystem(name, subsys=timeseries_comp)

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
