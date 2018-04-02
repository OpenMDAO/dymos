from __future__ import division, print_function, absolute_import

from six import iteritems
import numpy as np

from dymos.phases.phase_base import PhaseBase
from dymos.phases.components import BoundaryConstraintComp, EndpointConditionsComp, ContinuityComp
from dymos.ode_options import ODEOptions
from dymos.glm.dynamic_interp_comp import DynamicInterpComp
from dymos.glm.ozone.ode_integrator import ODEIntegrator
from dymos.glm.ozone.methods_list import method_classes
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
            normalized_times=self._norm_times
        )

        self.add_subsystem(
            'ozone', ode_int,
            promotes_outputs=[
                ('IC_state:{0}'.format(state), 'states:{0}'.format(state))
                for state in self.state_options
            ] + [
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
            normalized_times=self._norm_times
        )

        self.add_subsystem(
            'ozone', ode_int,
            promotes_outputs=[
                ('IC_state:{0}'.format(state), 'states:{0}'.format(state))
                for state in self.state_options
            ] + [
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
                output[:-1, ...] = control_comp._outputs['controls:{0}'.format(var)][::2, ...]
                output[-1, ...] = control_comp._outputs['controls:{0}'.format(var)][-1, ...]
            else:
                raise NotImplementedError()

        elif var_type == 'input_control':
            parameter_comp = self.input_controls
            if self.control_options[var]['dynamic']:
                output = np.zeros((num_segments + 1,) + self.control_options[var]['shape'])
                output[:-1, ...] = control_comp._outputs['controls:{0}'.format(var)][::2, ...]
                output[-1, ...] = control_comp._outputs['controls:{0}'.format(var)][-1, ...]
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
        pass

    def _setup_boundary_constraints(self):
        """
        Adds BoundaryConstraintComp if necessary and issues appropriate connections.
        """
        transcription = self.metadata['transcription']
        bc_comp = None

        if self._boundary_constraints:
            bc_comp = self.add_subsystem('boundary_constraints', subsys=BoundaryConstraintComp())

        for var, options in iteritems(self._boundary_constraints):
            con_name = options['constraint_name']
            con_units = options.get('units', None)
            con_shape = options.get('shape', (1,))

            # Determine the path to the variable which we will be constraining
            var_type = self._classify_var(var)

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
                # Failed to find variable, assume it is in the RHS
                if transcription == 'gauss-lobatto':
                    constraint_path = 'rhs_disc.{0}'.format(var)
                elif transcription == 'radau-ps':
                    constraint_path = 'rhs_all.{0}'.format(var)
                else:
                    raise ValueError('Invalid transcription')

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
