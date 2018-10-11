from __future__ import division, print_function, absolute_import

from collections import Iterable

import numpy as np
from dymos.phases.components import EndpointConditionsComp
from dymos.phases.phase_base import PhaseBase
from dymos.phases.grid_data import GridData
from openmdao.api import IndepVarComp, ParallelGroup
from six import iteritems

from .components.segment.explicit_segment import ExplicitSegment
from ...utils.rk_methods import rk_methods
from ...utils.misc import CoerceDesvar, get_rate_units
from ...utils.constants import INF_BOUND


class ExplicitPhase(PhaseBase):
    """
    ExplicitPhase provides explicit time stepping multiple shooting phases.

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
    def __init__(self, num_segments, transcription_order=3, num_steps=10, segment_ends=None,
                 compressed=True, **kwargs):
        kwgs = kwargs.copy()
        kwgs.update({'num_segments': num_segments, 'transcription_order': transcription_order,
                    'segment_ends': segment_ends, 'num_steps': num_steps, 'compressed': compressed})

        super(ExplicitPhase, self).__init__(**kwgs)

        # Pluck out the kwargs needed to initialize grid_data, potentially needed prior to setup.
        num_segments = num_segments
        transcription_order = transcription_order
        segment_ends = segment_ends
        compressed = compressed

        self.grid_data = GridData(num_segments=num_segments, transcription='explicit',
                                  transcription_order=transcription_order,
                                  num_steps_per_segment=num_steps, segment_ends=segment_ends,
                                  compressed=compressed)

    def initialize(self):
        super(ExplicitPhase, self).initialize()
        self.options['transcription'] = 'explicit'
        self.options.declare('num_steps', default=10, types=(int, Iterable),
                             desc='Number of steps to take within each segment.')
        self.options.declare('method', default='rk4', values=('rk4',),
                             desc='The integration method used within the explicit phase.')

    def setup(self):
        super(ExplicitPhase, self).setup()

        transcription = self.options['transcription']

        num_opt_controls = len([name for (name, options) in iteritems(self.control_options)
                                if options['opt']])

        num_controls = len(self.control_options)

        indep_controls = ['indep_controls'] if num_opt_controls > 0 else []
        design_params = ['design_params'] if self.design_parameter_options else []
        input_params = ['input_params'] if self.input_parameter_options else []
        control_interp_comp = ['control_interp_comp'] if num_controls > 0 else []

        order = self._time_extents + indep_controls + \
            input_params + design_params + \
            ['indep_states', 'time'] + control_interp_comp + ['indep_jumps', 'endpoint_conditions']

        order = order + ['segments']

        # if self.grid_data.num_segments > 1:
        #     order.append('continuity_comp')
        if getattr(self, 'boundary_constraints', None) is not None:
            order.append('boundary_constraints')
        if getattr(self, 'path_constraints', None) is not None:
            order.append('path_constraints')
        self.set_order(order)

    def _setup_time(self):
        comps = super(ExplicitPhase, self)._setup_time()
        gd = self.grid_data

        for iseg in range(gd.num_segments):
            i1, i2 = gd.subset_segment_indices['all'][iseg, :]
            seg_idxs = gd.subset_node_indices['all'][i1:i2]
            seg_end_idxs = seg_idxs[[0, -1]]
            self.connect('time', 'seg_{0}.seg_t0_tf'.format(iseg),
                         src_indices=seg_end_idxs)
        return comps

    def _setup_rhs(self):
        gd = self.grid_data

        segments_group = self.add_subsystem('segments', subsys=ParallelGroup(),
                                            promotes_inputs=['*'], promotes_outputs=['*'])

        for iseg in range(gd.num_segments):

            segment_i = ExplicitSegment(index=iseg,
                                        grid_data=self.grid_data,
                                        num_steps=self.options['num_steps'],
                                        method='rk4',
                                        ode_class=self.options['ode_class'],
                                        ode_init_kwargs=self.options['ode_init_kwargs'],
                                        time_options=self.time_options,
                                        state_options=self.state_options,
                                        control_options=self.control_options,
                                        design_parameter_options=self.design_parameter_options,
                                        input_parameter_options=self.input_parameter_options)

            segments_group.add_subsystem('seg_{0}'.format(iseg),
                                         subsys=segment_i)

    def _setup_states(self):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        gd = self.grid_data
        num_state_input_nodes = gd.subset_num_nodes['state_input']

        indep = IndepVarComp()
        for name, options in iteritems(self.state_options):
            indep.add_output(name='states:{0}'.format(name),
                             shape=(num_state_input_nodes, np.prod(options['shape'])),
                             units=options['units'])

            for iseg in range(gd.num_segments):
                i1, i2 = gd.subset_segment_indices['all'][iseg, :]
                self.connect('states:{0}'.format(name),
                             'seg_{0}.initial_states:{1}'.format(iseg, name),
                             src_indices=gd.subset_node_indices['state_disc'][iseg])

        self.add_subsystem('indep_states', indep, promotes_outputs=['*'])

        # Add the initial state values as design variables, if necessary

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            if options['opt']:
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

                    self.add_design_var(name='states:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar_option('scaler'),
                                        adder=coerce_desvar_option('adder'),
                                        ref0=coerce_desvar_option('ref0'),
                                        ref=coerce_desvar_option('ref'),
                                        indices=desvar_indices)

    def _setup_controls(self):
        super(ExplicitPhase, self)._setup_controls()

        gd = self.grid_data

        for name, options in iteritems(self.control_options):

            for iseg in range(gd.num_segments):
                i1, i2 = gd.subset_segment_indices['control_input'][iseg]

                self.connect('controls:{0}'.format(name),
                             'seg_{0}.disc_controls:{1}'.format(iseg, name),
                             src_indices=np.arange(i1, i1+i2, dtype=int))

    def _setup_defects(self):
        """
        Setup the Collocation and Continuity components as necessary.
        """
        pass

    def _setup_endpoint_conditions(self):

        jump_comp = self.add_subsystem('indep_jumps', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        jump_comp.add_output('initial_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the start of the phase')

        jump_comp.add_output('final_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the end of the phase')

        endpoint_comp = EndpointConditionsComp(time_options=self.time_options,
                                               state_options=self.state_options,
                                               control_options=self.control_options)

        self.connect('time', 'endpoint_conditions.values:time')

        self.connect('initial_jump:time',
                     'endpoint_conditions.initial_jump:time')

        self.connect('final_jump:time',
                     'endpoint_conditions.final_jump:time')

        promoted_list = ['time--', 'time-+', 'time+-', 'time++']

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
                         'endpoint_conditions.values:{0}'.format(state_name))

            self.connect('initial_jump:{0}'.format(state_name),
                         'endpoint_conditions.initial_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(state_name),
                         'endpoint_conditions.final_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            promoted_list += ['states:{0}--'.format(state_name),
                              'states:{0}-+'.format(state_name),
                              'states:{0}+-'.format(state_name),
                              'states:{0}++'.format(state_name)]

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

            self.connect('control_interp_comp.control_values:{0}'.format(control_name),
                         'endpoint_conditions.values:{0}'.format(control_name))

            self.connect('initial_jump:{0}'.format(control_name),
                         'endpoint_conditions.initial_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(control_name),
                         'endpoint_conditions.final_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            promoted_list += ['controls:{0}--'.format(control_name),
                              'controls:{0}-+'.format(control_name),
                              'controls:{0}+-'.format(control_name),
                              'controls:{0}++'.format(control_name)]

        self.add_subsystem(name='endpoint_conditions', subsys=endpoint_comp,
                           promotes_outputs=promoted_list)

    def _setup_path_constraints(self):
        pass

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
        template = 'seg_{0}.stage_ode.{1}'
        num_stages = rk_methods[self.options['method']]['num_stages']

        if name in self.ode_options._parameters:
            ode_tgts = self.ode_options._parameters[name]['targets']

            for i in range(self.grid_data.num_segments):
                num_steps = self.grid_data.num_steps_per_segment[i]
                num_nodes = num_stages * num_steps
                src_idxs = [0] * num_nodes
                connection_info.append(([template.format(i, t) for t in ode_tgts], src_idxs))

        return connection_info

    def _get_boundary_constraint_src(self, var, loc):
        # Determine the path to the variable which we will be constraining
        gd = self.grid_data
        time_units = self.time_options['units']
        var_type = self._classify_var(var)

        src_seg = 'seg_{0}'.format(0 if loc == 'initial' else self.grid_data.num_segments - 1)

        if var_type == 'time':
            shape = (1,)
            units = self.time_units
            linear = True
            constraint_path = '{0}.t_step'.format(src_seg)
        elif var_type == 'state':
            state_shape = self.state_options[var]['shape']
            state_units = self.state_options[var]['units']
            shape = state_shape
            units = state_units
            linear = True
            constraint_path =  '{0}.step_states:{0}'.format(src_seg, var)
        elif var_type in 'indep_control':
            control_shape = self.control_options[var]['shape']
            control_units = self.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'control_values:{0}'.format(var)
        elif var_type == 'input_control':
            control_shape = self.control_options[var]['shape']
            control_units = self.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'control_values:{0}'.format(var)
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
        elif var_type == 'control_rate':
            control_var = var[:-5]
            control_shape = self.control_options[control_var]['shape']
            control_units = self.control_options[control_var]['units']
            control_rate_units = get_rate_units(control_units, time_units, deriv=1)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        elif var_type == 'control_rate2':
            control_var = var[:-6]
            control_shape = self.control_options[control_var]['shape']
            control_units = self.control_options[control_var]['units']
            control_rate_units = get_rate_units(control_units, time_units, deriv=2)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        else:
            # Failed to find variable, assume it is in the RHS
            if self.options['transcription'] == 'gauss-lobatto':
                constraint_path = 'rhs_disc.{0}'.format(var)
            elif self.options['transcription'] == 'radau-ps':
                constraint_path = 'rhs_all.{0}'.format(var)
            else:
                raise ValueError('Invalid transcription')
            shape = None
            units = None
            linear = False

        # Build the correct src_indices regardless of shape
        size = int(np.prod(shape))

        if loc == 'initial':
            src_idxs = np.arange(size, dtype=int).reshape(shape)
        else:
            src_idxs = np.arange(-size, 0, dtype=int).reshape(shape)

        return constraint_path, src_idxs, shape, units, linear

    # def get_values(self, var, nodes='solution', units=None):
    #     """
    #     Retrieve the values of the given variable at the given
    #     subset of nodes.
    #
    #     Parameters
    #     ----------
    #     var : str
    #         The variable whose values are to be returned.  This may be
    #         the name 'time', the name of a state, control, or parameter,
    #         or the path to a variable in the ODEFunction of the phase.
    #     nodes : str
    #         The name of the node subset.
    #     units : str
    #         The units in which the values should be expressed.  Must be compatible
    #         with the corresponding units inside the phase.
    #
    #     Returns
    #     -------
    #     ndarray
    #         An array of the values at the requested node subset.  The
    #         node index is the first dimension of the ndarray.
    #     """
    #     gd = self.grid_data
    #     num_stages = rk_methods[self.options['method']]['s']
    #
    #     var_type = self._classify_var(var)
    #
    #     outputs = dict(self.list_outputs(explicit=True, values=True, units=True, shape=True, out_stream=None))
    #     inputs = dict(self.list_inputs(values=True, units=True, out_stream=None))
    #
    #     # print('\n'.join([o for o in outputs if 'states:y' in o]))
    #
    #     if units is not None:
    #         if not valid_units(units):
    #             raise ValueError('Units {0} is not a valid units identifier'.format(units))
    #
    #     path = '{0}.'.format(self.pathname) if self.pathname else ''
    #
    #     path_map = {'time': 'time.{0}',
    #                 'state': 'indep_states.states:{0}',
    #                 'indep_control': 'control_interp_comp.control_values:{0}',
    #                 'input_control': 'control_interp_comp.control_values:{0}',
    #                 'design_parameter': 'design_params.design_parameters:{0}',
    #                 'input_parameter': 'input_params.input_parameters:{0}_out',
    #                 'control_rate': 'control_interp_comp.control_rates:{0}',
    #                 'control_rate2': 'control_interp_comp.control_rates:{0}',
    #                 'ode': 'rhs_all.{0}'}
    #
    #     if var_type == 'state':
    #         var_path = path + path_map[var_type].format(var)
    #
    #         output_value = np.nan * np.ones(gd.subset_num_nodes['all'])
    #
    #         tmp = path + 'segments.seg_{0}.step_{1}.ycomp_stage_{2}.states:{3}_{2}'
    #
    #         # Populate the stage values
    #         stage_values = [[[[outputs[tmp.format(iseg, jstep, kstage, var)]['value']
    #                            for kstage in range(1, num_stages + 1)]
    #                           for jstep in range(self.grid_data.num_steps_per_segment[iseg])]
    #                          for iseg in range(self.grid_data.num_segments)]]
    #         stage_units = outputs[tmp.format(0, 0, 1, var)]['units']
    #         stage_values = np.asarray(stage_values).ravel()
    #         output_value[gd.subset_node_indices['stage_nodes']] = convert_units(stage_values,
    #                                                                             stage_units,
    #                                                                             units)
    #
    #         # Populate the step end values
    #         step_end_idxs = gd.subset_node_indices['step_ends'][1::2]
    #         for iseg in range(gd.num_segments):
    #             for jstep in range(self.grid_data.num_steps_per_segment[iseg]):
    #                 advance_units = outputs[path + 'segments.seg_{0}.step_{1}.advance.states:{2}_f'.format(iseg, jstep, var)]['units']
    #                 val = outputs[path + 'segments.seg_{0}.step_{1}.advance.states:{2}_f'.format(iseg, jstep, var)]['value']
    #                 output_value[step_end_idxs[jstep]] = convert_units(val, advance_units, units)
    #
    #     elif var_type in ('input_control', 'indep_control'):
    #         var_path = path + path_map[var_type].format(var)
    #         output_units = outputs[var_path]['units']
    #
    #         vals = outputs[var_path]['value']
    #         output_value = convert_units(vals, output_units, units)
    #
    #     elif var_type in ('design_parameter', 'input_parameter', 'traj_design_parameter',
    #                       'traj_input_parameter'):
    #         var_path = path + path_map[var_type].format(var)
    #         output_units = outputs[var_path]['units']
    #
    #         output_value = convert_units(outputs[var_path]['value'], output_units, units)
    #         output_value = np.repeat(output_value, gd.num_nodes, axis=0)
    #
    #     elif var_type == 'ode':
    #         rhs_all_outputs = dict(self.rhs_all.list_outputs(out_stream=None, values=True,
    #                                                          shape=True, units=True))
    #         prom2abs_all = self.rhs_all._var_allprocs_prom2abs_list
    #         abs_path_all = prom2abs_all['output'][var][0]
    #         output_value = rhs_all_outputs[abs_path_all]['value']
    #         output_units = rhs_all_outputs[abs_path_all]['units']
    #         output_value = convert_units(output_value, output_units, units)
    #     else:
    #         var_path = path + path_map[var_type].format(var)
    #         output_units = outputs[var_path]['units']
    #         output_value = convert_units(outputs[var_path]['value'], output_units, units)
    #
    #     # Always return a column vector
    #     if len(output_value.shape) == 1:
    #         output_value = np.reshape(output_value, (gd.num_nodes, 1))
    #
    #     return output_value[gd.subset_node_indices[nodes], ...]