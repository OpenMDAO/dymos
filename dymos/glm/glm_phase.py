from __future__ import division, print_function, absolute_import

from six import iteritems
import numpy as np

from dymos.phases.phase_base import PhaseBase
from dymos.phases.components import BoundaryConstraintComp, EndpointConditionsComp, ContinuityComp
from dymos.phases.components import GLMPathConstraintComp
from dymos.ode_options import ODEOptions
from dymos.glm.dynamic_interp_comp import DynamicInterpComp
from dymos.glm.ozone.ode_integrator import ODEIntegrator
from dymos.glm.ozone.methods_list import method_classes, get_method

from openmdao.api import IndepVarComp, NonlinearBlockGS


class GLMPhase(PhaseBase):

    def __init__(self, **kwargs):
        super(GLMPhase, self).__init__(**kwargs)

        self._fixed_states = set()
        self._dynamics = None
        self._norm_times = None
        self._node_indices = None
        self._segment_times = None

    def initialize(self):
        super(GLMPhase, self).initialize()
        # Optional metadata

        self.metadata.declare(
            'formulation', default='solver-based',
            values=['optimizer-based', 'solver-based', 'time-marching'],
            desc='Formulation for solving the ODE.')

        self.metadata.declare(
            'method_name', default='RK4', values=set(method_classes.keys()),
            desc='Scheme used to integrate the ODE.')
        self.metadata.declare(
            'num_timesteps', types=int,
            desc='Minimum number of timesteps, more may be used.')

        self.metadata['transcription'] = 'glm'

    def setup(self):
        super(GLMPhase, self).setup()

        num_opt_controls = len([name for (name, options) in iteritems(self.control_options)
                                if options['opt']])

        num_input_controls = len([name for (name, options) in iteritems(self.control_options)
                                  if not options['opt']])

        num_controls = num_opt_controls + num_input_controls

        if num_opt_controls > 0:
            indep_controls = ['indep_controls']
        if num_input_controls > 0:
            input_parameters = ['input_controls']
        if num_controls > 0:
            control_rate_comp = ['control_rate_comp']

        order = self._time_extents[:]

        order += ['time', 'indep_jumps', 'ozone']
        if self.control_options:
            order.insert(-1, 'control_rate_comp')
            if num_opt_controls > 0:
                order = ['indep_controls'] + order
            if num_input_controls > 0:
                order = ['input_controls'] + order
            if self._dynamics:
                order.insert(3, 'dynamic_interp')
                self.add_subsystem('dynamic_interp',
                                   DynamicInterpComp(grid_data=self.grid_data,
                                                     control_options=self.control_options,
                                                     time_units=self.time_options['units'],
                                                     normalized_times=self._norm_times,
                                                     segment_times=self._segment_times))

        if self._fixed_states:
            order = ['fixed_states'] + order
            if 'final' in self._fixed_states:
                order.append('final_balance')

        order.append('endpoint_conditions')

        continuity_comp_exists = False
        for control_name, options in iteritems(self.control_options):
            if options['dynamic'] and options['continuity']:
                continuity_comp_exists = True
        if continuity_comp_exists:
            order.append('continuity_constraint')

        if len(self._boundary_constraints) > 0:
            order.append('boundary_constraints')

        if len(self._path_constraints) > 0:
            order.append('path_constraints')

        self.set_order(order)

    def _setup_rhs(self):
        pass

    def _setup_defects(self):
        grid_data = self.grid_data
        time_units = self.time_options['units']
        num_segment_boundaries = grid_data.num_segments - 1

        if num_segment_boundaries > 0:
            # Continuity Constraints
            continuity_comp = ContinuityComp(grid_data=grid_data,
                                             state_options=self.state_options,
                                             control_options=self.control_options,
                                             time_units=time_units,
                                             enforce_state_continuity=False)

            # State continuity is guaranteed by the ODE solver

            continuity = False

            for name, options in iteritems(self.control_options):
                if options['opt'] and options['dynamic']:
                    self.connect(
                        'controls:{0}'.format(name),
                        'continuity_constraint.controls:{}'.format(name),
                        src_indices=grid_data.subset_node_indices['disc'])
                    continuity = True
                    if options['rate_continuity']:
                        continuity = True
                        self.connect('control_rates:{0}_rate'.format(name),
                                     'continuity_constraint.control_rates:{}_rate'.format(name),
                                     src_indices=grid_data.subset_node_indices['disc'])
                    if options['rate2_continuity']:
                        continuity = True
                        self.connect('control_rates:{0}_rate'.format(name),
                                     'continuity_constraint.control_rates:{}_rate'.format(name),
                                     src_indices=grid_data.subset_node_indices['disc'])

            if continuity:
                self.add_subsystem('continuity_constraint', continuity_comp)

    def _setup_time(self):
        comps = super(GLMPhase, self)._setup_time()
        grid_data = self.grid_data

        # TODO: Build interpolant rather than forcing a timestep.
        min_steps = self.metadata['num_segments']
        max_h = 1. / min_steps
        # node_times = (grid_data.node_ptau + 1.0) / 2.0
        node_indices = []
        normalized_times = []

        current_time = 0.
        i = 0
        segment_times = []
        for iseg1, iseg2 in grid_data.segment_indices:
            node_times = (grid_data.node_ptau[iseg1:iseg2] + 1.0) / 2.0
            i_old = i
            for node_time in node_times:
                while node_time - current_time > max_h:
                    current_time += max_h
                    normalized_times.append(current_time)
                    i += 1
                current_time = node_time
                normalized_times.append(current_time)
                node_indices.append(i)
                i += 1
            i_new = i
            segment_times.append([i_old, i_new])

        self._norm_times = np.zeros(2 * self.metadata['num_segments'])
        self._norm_times[0::2] = grid_data.segment_ends[:-1]
        self._norm_times[1::2] = grid_data.segment_ends[1:]
        self._norm_times = (self._norm_times - self._norm_times[0]) \
            / (self._norm_times[-1] - self._norm_times[0])

        self._segment_times = segment_times = []
        for iseg1, iseg2 in grid_data.segment_indices:
            segment_times.append([iseg1, iseg2])

        self.connect('time', 'ozone.initial_time', src_indices=[0])
        self.connect('time', 'ozone.final_time', src_indices=[-1])

        return comps

    def _setup_controls(self):
        super(GLMPhase, self)._setup_controls()

        for control, opts in iteritems(self.control_options):
            control_src_name = 'controls:{0}'.format(control) if opts['opt'] else \
                'controls:{0}_out'.format(control)
            if opts['dynamic']:
                self._dynamics = True
                self.connect(control_src_name,
                             'dynamic_interp.dynamic_nodes:{0}'.format(control))
                self.connect('dynamic_interp.dynamic_ts:{0}'.format(control),
                             'ozone.dynamic_parameter:{0}'.format(control))

    def _setup_sand(self):
        ode_int = ODEIntegrator(
            self.metadata['ode_class'],
            self.metadata['formulation'],
            self.metadata['method_name'],
            normalized_times=self._norm_times,
            state_options=self.state_options,
        )

        self.add_subsystem(
            'ozone', ode_int,
            promotes_outputs=[
                ('state:{0}'.format(state), 'out_states:{0}'.format(state))
                for state in self.state_options
            ],
            promotes_inputs=[
                'initial_condition:{0}'.format(state)
                for state in self.state_options
            ],
        )

    def _setup_mdf(self):
        ode_int = ODEIntegrator(
            self.metadata['ode_class'],
            self.metadata['formulation'],
            self.metadata['method_name'],
            normalized_times=self._norm_times,
            state_options=self.state_options,
        )

        self.add_subsystem(
            'ozone', ode_int,
            promotes_outputs=[
                ('state:{0}'.format(state), 'out_states:{0}'.format(state))
                for state in self.state_options
            ],
            promotes_inputs=[
                'initial_condition:{0}'.format(state)
                for state in self.state_options
            ])

    def _setup_states(self):
        formulation = self.metadata['formulation']
        if formulation == 'optimizer-based':
            self._setup_sand()
        elif formulation == 'solver-based' or formulation == 'time-marching':
            self._setup_mdf()
        else:
            raise ValueError('Unknown formulation: {}'.format(formulation))

    def get_values(self, var, nodes=None):
        """
        Retrieve the values of the given variable at the given
        subset of nodes.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.  This may be
            the name 'time', the name of a state, control, or parameter,
            or the path to a variable in the ODEFunction of the phase.
        nodes : str
            The name of the node subset, one of 'disc', 'col', 'None'.
            This option does not apply to GLMPhase. The default is 'None'.

        Returns
        -------
        ndarray
            An array of the values at the requested node subset.  The
            node index is the first dimension of the ndarray.
        """
        if nodes is not None:
            raise ValueError('With GLMPhase, nodes=None is the only valid option.')

        gd = self.grid_data

        var_type = self._classify_var(var)

        num_segments = self.metadata['num_segments']

        if var_type == 'time':
            output = np.zeros((num_segments + 1, 1))
            time_comp = self.time
            output[:-1, 0] = time_comp._outputs[var][::2]
            output[-1, 0] = time_comp._outputs[var][-1]

        elif var_type == 'state':
            output = np.zeros((num_segments + 1,) + self.state_options[var]['shape'])
            state_vals = self._outputs['out_states:{0}'.format(var)]

            output[:-1, ...] = state_vals[::2, ...]
            output[-1, ...] = state_vals[-1, ...]

        elif var_type == 'indep_control':
            control_comp = self.indep_controls
            if self.control_options[var]['dynamic']:
                output = np.zeros((num_segments + 1,) + self.control_options[var]['shape'])
                output[:-1, ...] = self._outputs['controls:{0}'.format(var)][::2, ...]
                output[-1, ...] = self._outputs['controls:{0}'.format(var)][-1, ...]
            else:
                raise NotImplementedError()

        elif var_type == 'input_control':
            parameter_comp = self.input_controls
            if self.control_options[var]['dynamic']:
                output = np.zeros((num_segments + 1,) + self.control_options[var]['shape'])
                output[:-1, ...] = \
                    self._outputs['input_controls.controls:{0}_out'.format(var)][::2, ...]
                output[-1, ...] = \
                    self._outputs['input_controls.controls:{0}_out'.format(var)][-1, ...]
            else:
                raise NotImplementedError()

        elif var_type == 'control_rate':
            control_rate_comp = self.control_rate_comp
            output = np.zeros((num_segments + 1,) + self.control_options[var]['shape'])
            output[:-1, ...] = \
                control_rate_comp._outputs['control_rates:{0}_rate'.format(var)][::2, ...]
            output[-1, ...] = \
                control_rate_comp._outputs['control_rates:{0}_rate'.format(var)][-1, ...]

        elif var_type == 'parameter_rate':
            control_rate_comp = self.control_rate_comp
            output = np.zeros((num_segments + 1,) + self.control_options[var]['shape'])
            output[:-1, ...] = \
                control_rate_comp._outputs['parameter_rates:{0}_rate'.format(var)][::2, ...]
            output[-1, ...] = \
                control_rate_comp._outputs['parameter_rates:{0}_rate'.format(var)][-1, ...]

        elif var_type == 'rhs':
            raise NotImplementedError()

        return output

    def _setup_path_constraints(self):
        """
        Add a path constraint component if necessary and issue appropriate connections as
        part of the setup stack.
        """
        num_timesteps = self.metadata['num_segments']
        num_stages = get_method(self.metadata['method_name']).num_stages

        for var in self._path_constraints:
            var_type = self._classify_var(var)

            self._path_constraints[var]['type_'] = var_type

        path_comp = None
        if self._path_constraints:
            path_comp = GLMPathConstraintComp(num_timesteps=num_timesteps, num_stages=num_stages)
            self.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(self._path_constraints):
            con_units = options.get('units', None)
            con_name = options['constraint_name']

            # Determine the path to the variable which we will be constraining
            # This is more complicated for path constraints since, for instance,
            # a single state variable has two sources which must be connected to
            # the path component.
            var_type = self._classify_var(var)

            if var_type == 'state':
                state_shape = self.state_options[var]['shape']
                state_units = self.state_options[var]['units']
                options['shape'] = state_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = False
                self.connect(src_name='out_states:{0}'.format(var),
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'indep_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'controls:{0}'.format(var)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'input_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'input_controls:{0}_out'.format(var)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate':
                control_name = var[:-5]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            elif var_type == 'control_rate2':
                control_name = var[:-6]
                control_shape = self.control_options[control_name]['shape']
                control_units = self.control_options[control_name]['units']
                options['shape'] = control_shape
                options['units'] = control_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}_rate2'.format(control_name)
                self.connect(src_name=constraint_path,
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            else:
                # Failed to find variable, assume it is in the RHS
                options['linear'] = False
                self.connect(src_name='ozone.integration_group.ode_comp.{0}'.format(var),
                             tgt_name='path_constraints.all_values:{0}'.format(con_name))

            kwargs = options.copy()
            if var_type == 'control_rate':
                kwargs['units'] = get_rate_units(options['units'],
                                                 self.time_options['units'],
                                                 deriv=1)
            elif var_type == 'control_rate2':
                kwargs['units'] = get_rate_units(options['units'],
                                                 self.time_options['units'],
                                                 deriv=2)
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, var_type, **kwargs)

    def _setup_boundary_constraints(self):
        """
        Adds BoundaryConstraintComp if necessary and issues appropriate connections.
        """
        transcription = self.metadata['transcription']
        formulation = self.metadata['formulation']
        bc_comp = None

        num_timesteps = self.metadata['num_segments']
        num_stages = get_method(self.metadata['method_name']).num_stages

        if self._boundary_constraints:
            bc_comp = self.add_subsystem('boundary_constraints', subsys=BoundaryConstraintComp())

        for var, options in iteritems(self._boundary_constraints):
            con_name = options['constraint_name']
            con_units = options.get('units', None)
            con_shape = options.get('shape', (1,))

            # Determine the path to the variable which we will be constraining
            var_type = self._classify_var(var)

            custom_var = False
            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = self.time_options['units'] if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'time'
            elif var_type == 'state':
                state_shape = self.state_options[var]['shape']
                state_units = self.state_options[var]['units']
                options['shape'] = state_shape if con_shape is None else con_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'out_states:{0}'.format(var)
            elif var_type == 'indep_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'controls:{0}'.format(var)
            elif var_type == 'input_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'controls:{0}_out'.format(var)
            elif var_type == 'control_rate':
                control_var = var[:-5]
                control_shape = self.control_options[control_var]['shape']
                control_units = self.control_options[control_var]['units']
                control_rate_units = get_rate_units(control_units,
                                                    self.time_options['units'],
                                                    deriv=1)
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_rate_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}'.format(var)
            elif var_type == 'control_rate2':
                control_var = var[:-6]
                control_shape = self.control_options[control_var]['shape']
                control_units = self.control_options[control_var]['units']
                control_rate_units = get_rate_units(control_units,
                                                    self.time_options['units'],
                                                    deriv=2)
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_rate_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}'.format(var)
            else:
                custom_var = True

                # Failed to find variable, assume it is in the RHS
                if formulation == 'optimizer-based' or formulation == 'solver-based':
                    constraint_path = 'ozone.integration_group.ode_comp.{0}'.format(var)
                else:
                    raise NotImplementedError()

                options['shape'] = con_shape
                options['units'] = con_units

            if 'initial' in options:
                options['initial']['units'] = options['units']
                bc_comp._add_initial_constraint(con_name,
                                                **options['initial'])
            if 'final' in options:
                options['final']['units'] = options['units']
                bc_comp._add_final_constraint(con_name,
                                              **options['final'])

            # Build the correct src_indices regardless of shape
            if custom_var:
                indices = np.arange(num_timesteps * num_stages * np.prod(con_shape)).reshape(
                    (num_timesteps, num_stages,) + con_shape)
                src_idxs_initial = indices[0, 0].flatten()
                src_idxs_final = indices[-1, 0].flatten()
                src_idxs = np.concatenate([src_idxs_initial, src_idxs_final])
            else:
                size = np.prod(options['shape'])
                src_idxs_initial = np.arange(size, dtype=int).reshape(options['shape'])
                src_idxs_final = np.arange(-size, 0, dtype=int).reshape(options['shape'])
                src_idxs = np.stack((src_idxs_initial, src_idxs_final))

            self.connect(constraint_path,
                         'boundary_constraints.boundary_values:{0}'.format(con_name),
                         src_indices=src_idxs, flat_src_indices=True)

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

            self.connect('out_states:{0}'.format(state_name),
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

            if options['opt']:
                suffix = ''
            else:
                suffix = '_out'

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

            self.connect('controls:{0}{1}'.format(control_name, suffix),
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

    def add_objective(self, name, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False, simul_coloring=None, simul_map=None):
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
        simul_coloring : ndarray or list of int
            An array or list of integer color values.  Must match the size of the
            objective variable.
        simul_map : dict
            Mapping of this response to each design variable where simultaneous derivs will
            be used.  Each design variable entry is another dict keyed on color, and the values
            in the color dict are tuples of the form (resp_idxs, color_idxs).
        """
        var_type = self._classify_var(name)

        # Determine the path to the variable
        if var_type == 'time':
            obj_path = 'ozone.times'
        elif var_type == 'state':
            obj_path = 'ozone.state:{0}'.format(name)
        elif var_type == 'indep_control':
            obj_path = 'ozone.dynamic_parameter:{0}'.format(name)
        elif var_type == 'input_control':
            obj_path = 'ozone.dynamic_parameter:{0}'.format(name)
        elif var_type == 'control_rate':
            raise NotImplementedError()
            # control_name = name[:-5]
            # obj_path = 'control_rates:{0}_rate'.format(control_name)
        elif var_type == 'control_rate2':
            raise NotImplementedError()
            # control_name = name[:-6]
            # obj_path = 'control_rates:{0}_rate2'.format(control_name)
        else:
            raise NotImplementedError()
            # Failed to find variable, assume it is in the RHS
            # obj_path = 'rhs_disc.{0}'.format(name)

        super(GLMPhase, self)._add_objective(obj_path, loc=loc, index=index, shape=shape,
                                             ref=ref, ref0=ref0, adder=adder,
                                             scaler=scaler,
                                             parallel_deriv_color=parallel_deriv_color,
                                             vectorize_derivs=vectorize_derivs,
                                             simul_coloring=simul_coloring,
                                             simul_map=simul_map)

    def set_values(self, var, val, nodes=None, kind='linear', axis=0):
        """
        Retrieve the values of the given variable at the given
        subset of nodes.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.  This may be
            the name 'time', the name of a state, control, or parameter,
            or the path to a variable in the ODE system of the phase.
        val : ndarray
            Array of time/control/state/parameter values.
        nodes : str
            The name of the node subset, one of 'disc', 'col', 'None'.
            This option does not apply to GLMPhase. The default is 'None'.
        kind : str
            Specifies the kind of interpolation, as per the scipy.interpolate package.
            One of ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.
        axis : int
            Specifies the axis along which interpolation should be performed.  Default is
            the first axis (0).
        """
        var_type = self._classify_var(var)
        formulation = self.metadata['formulation']
        num_stages = get_method(self.metadata['method_name']).num_stages

        if var_type == 'time':
            interpolated_values = self.interpolate(xs=val, nodes=nodes, kind=kind, axis=axis)
        else:
            interpolated_values = self.interpolate(ys=val, nodes=nodes, kind=kind, axis=axis)

        if var_type == 'state':
            self._outputs['ozone.initial_condition:{}'.format(var)] = interpolated_values[0]
            if formulation == 'optimizer-based':
                name = 'ozone.integration_group.desvars_comp.Y:{}'.format(var)
                self._outputs[name] = \
                    np.einsum('i...,j->ij...', interpolated_values[1:], np.ones(num_stages))
            elif formulation == 'solver-based':
                name = 'ozone.integration_group.vectorized_stagestep_comp.Y_out:{}'.format(var)
                self._outputs[name] = \
                    np.einsum('i...,j->ij...', interpolated_values[1:], np.ones(num_stages))
        elif var_type == 'indep_control':
            self._outputs['controls:{}'.format(var)] = interpolated_values
        elif var_type == 'input_control':
            self._outputs['input_controls.controls:{0}_out'.format(var)] = interpolated_values

    def add_control(self, name, val=0.0, units=0, dynamic=True, opt=True, lower=None, upper=None,
                    fix_initial=False, fix_final=False,
                    scaler=None, adder=None, ref=None, ref0=None, continuity=None,
                    rate_continuity=None, rate2_continuity=None,
                    rate_param=None, rate2_param=None):
        """
        Declares that a parameter of the ODE is to potentially be used as an optimal control.

        Parameters
        ----------
        name : str
            Name of the controllable parameter in the ODE.
        val : float or ndarray
            Default value of the control at all nodes.  If val scalar and the control
            is dynamic it will be broadcast.
        units : str or None or 0
            Units in which the control variable is defined.  If 0, use the units declared
            for the parameter in the ODE.
        dynamic : bool
            If True (default) this is a dynamic control, the values provided correspond to
            the number of nodes in the phase.  If False, this is a static control, sized (1,),
            and that value is broadcast to all nodes within the phase.
        opt : bool
            If True (default) the value(s) of this control will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False, the values of this control will exist in
            'phase_name.input_controls.controls:control_name', where it may be connected to
            external sources if desired.
        lower : float or ndarray
            The lower bound of the control at the nodes of the phase.
        upper : float or ndarray
            The upper bound of the control at the nodes of the phase.
        scaler : float or ndarray
            The scaler of the control value at the nodes of the phase.
        adder : float or ndarray
            The adder of the control value at the nodes of the phase.
        ref0 : float or ndarray
            The zero-reference value of the control at the nodes of the phase.
        ref : float or ndarray
            The unit-reference value of the control at the nodes of the phase
        contiuity : bool or None
            True if continuity in the value of the control is desired at the segment bounds.
            See notes about default values for continuity.
        rate_continuity : bool or None
            True if continuity in the rate of the control is desired at the segment bounds.
            See notes about default values for continuity.
        rate_param : None or str
            The name of the parameter in the ODE to which the first time-derivative
            of the control value is connected.
        rate2_param : None or str
            The name of the parameter in the ODE to which the second time-derivative
            of the control value is connected.

        Notes
        -----
        If continuity is None or rate continuity is None, the default value for
        continuity is True and rate continuity of False.

        The default value of continuity and rate continuity for input controls (opt=False)
        is False.

        The user may override these defaults by specifying them as True or False.

        """
        if rate_continuity:
            raise NotImplementedError(
                'GLMPhase does not have support for nonlinear controls within segments ' +
                'so rate_continuity cannot be enforced.')
        if rate2_continuity:
            raise NotImplementedError(
                'GLMPhase does not have support for nonlinear controls within segments ' +
                'so rate2_continuity cannot be enforced.')

        super(GLMPhase, self).add_control(
            name, val=val, units=units, dynamic=dynamic, opt=opt, lower=lower, upper=upper,
            fix_initial=fix_initial, fix_final=fix_final, scaler=scaler, adder=adder,
            ref=ref, ref0=ref0, continuity=continuity, rate_continuity=None,
            rate2_continuity=None, rate_param=rate_param, rate2_param=rate2_param)
