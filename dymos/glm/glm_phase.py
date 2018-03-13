from __future__ import division, print_function, absolute_import

from six import iteritems
import numpy as np

from dymos.phases.phase_base import PhaseBase
from dymos.phases.components.continuity_comp import ContinuityComp
from dymos.ode_options import ODEOptions
from dymos.glm.dynamic_interp_comp import DynamicInterpComp
from dymos.glm.equality_constraint_comp import EqualityConstraintComp
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
            'formulation', default='solver-based', values=['optimizer-based', 'solver-based', 'time-marching'],
            desc='Formulation for solving the ODE.')

        self.metadata.declare(
            'method_name', default='RK4', values=set(method_classes.keys()),
            desc='Scheme used to integrate the ODE.')
        self.metadata.declare(
            'num_timesteps', types=int,
            desc='Minimum number of timesteps, more may be used.')

        self.metadata['transcription'] = 'glm'

    def setup(self):
        self.metadata['ode_class'].ode_options = self._to_ozone_ode(
            self.metadata['ode_class'].ode_options)
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
        order.append('continuity_constraint')

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
                                             time_units=time_units)

            # State continuity is guaranteed by the ODE solver

            continuity = False

            for name, options in iteritems(self.control_options):
                if options['opt'] and options['dynamic']:
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

        self._norm_times = np.array(normalized_times)
        self._node_indices = node_indices
        self._segment_times = segment_times

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

        self.add_subsystem('ozone', ode_int,
                           promotes_outputs=[('state:{0}'.format(state), 'states:{0}'.format(state))
                                             for state in self.state_options],
                           promotes_inputs=['initial_condition:{0}'.format(state)
                                            for state in self.state_options]
                           )

        fixed_states = IndepVarComp()
        final_comp = EqualityConstraintComp()
        any_initial = False
        any_final = False
        for state, opts in iteritems(self.state_options):
            if opts['opt']:
                any_initial = True
                fixed_states.add_output('initial_condition:{0}'.format(state),
                                        shape=opts['shape'],
                                        units=opts['units'])
                if not opts['fix_initial']:
                    lb, ub = np.finfo(float).min, np.finfo(float).max
                    if opts['lower'] is not None:
                        lb = opts['lower']
                    if opts['upper'] is not None:
                        ub = opts['upper']

                    if opts['initial_bounds'] is not None:
                        lb, ub = opts['initial_bounds']

                    self.add_design_var(name='initial_condition:{0}'.format(state),
                                        lower=lb,
                                        upper=ub,
                                        scaler=opts['scaler'],
                                        adder=opts['adder'],
                                        ref0=opts['ref0'],
                                        ref=opts['ref']
                                        )
                if opts['fix_final']:
                    any_final = True
                    final_comp.add_balance('{0}'.format(state),
                                           eq_units=opts['units'])
                    fixed_states.add_output('final_condition:{0}'.format(state),
                                            shape=opts['shape'],
                                            units=opts['units'])
                    self.connect('final_condition:{0}'.format(state),
                                 'final_balance.lhs:{0}'.format(state))
                    self.connect('states:{0}'.format(state),
                                 'final_balance.rhs:{0}'.format(state),
                                 src_indices=list(-1-np.arange(np.prod(opts['shape']))))
        if any_initial:
            self.add_subsystem('fixed_states', fixed_states,
                               promotes=['*'])
            self._fixed_states.add('initial')

        if any_final:
            self.add_subsystem('final_balance', final_comp)
            self._fixed_states.add('final')

    def _setup_mdf(self):
        ode_int = ODEIntegrator(
            self.metadata['ode_class'],
            self.metadata['formulation'],
            self.metadata['method_name'],
            normalized_times=self._norm_times
        )

        self.add_subsystem('ozone', ode_int,
                           promotes_outputs=[('state:{0}'.format(state), 'states:{0}'.format(state))
                                             for state in self.state_options],
                           promotes_inputs=['initial_condition:{0}'.format(state)
                                            for state in self.state_options])

        self.nonlinear_solver = NonlinearBlockGS(iprint=2, maxiter=40, atol=1e-14, rtol=1e-12)
        # self.linear_solver = LinearBlockGS(iprint=2, maxiter=40, atol=1e-14, rtol=1e-12)

        fixed_states = IndepVarComp()
        final_comp = EqualityConstraintComp()
        any_initial = False
        any_final = False
        for state, opts in iteritems(self.state_options):
            if opts['opt']:
                any_initial = True
                fixed_states.add_output('initial_condition:{0}'.format(state),
                                        shape=opts['shape'],
                                        units=opts['units'])

                if opts['fix_final']:
                    any_final = True
                    fixed_states.add_output('final_condition:{0}'.format(state),
                                            shape=opts['shape'],
                                            units=opts['units'])
                    final_comp.add_balance('{0}'.format(state),
                                           eq_units=opts['units'])
                    self.connect('final_condition:{0}'.format(state),
                                 'final_balance.lhs:{0}'.format(state))
                    self.connect('states:{0}'.format(state),
                                 'final_balance.rhs:{0}'.format(state),
                                 src_indices=list(-1-np.arange(np.prod(opts['shape']))))

        if any_initial:
            self.add_subsystem('fixed_states', fixed_states,
                               promotes=['*'])
            self._fixed_states.add('initial')

        if any_final:
            self.add_subsystem('final_balance', final_comp)
            self._fixed_states.add('final')

    def _setup_states(self):
        formulation = self.metadata['formulation']
        if formulation == 'optimizer-based':
            self._setup_sand()
        elif formulation == 'solver-based' or formulation == 'time-marching':
            self._setup_mdf()
        else:
            raise ValueError('Unknown formulation: {}'.format(formulation))

    def get_values(self, var, nodes='all'):
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
            The name of a node subset, one of 'disc', 'col', 'all'.
            The default is 'all'.

        Returns
        -------
        ndarray
            An array of the values at the requested node subset.  The
            node index is the first dimension of the ndarray.
        """
        gd = self.grid_data

        var_type = self._classify_var(var)

        if var_type == 'time':
            output = np.zeros((self.grid_data.num_nodes, 1))
            time_comp = self.time
            output[:, 0] = time_comp._outputs[var]

        elif var_type == 'state':
            output = np.zeros((gd.num_nodes,) + self.state_options[var]['shape'])
            state_vals = self._outputs['states:{0}'.format(var)]

            output[:, ...] = state_vals[self._node_indices, ...]

        elif var_type == 'indep_control':
            control_comp = self.indep_controls
            if self.control_options[var]['dynamic']:
                output = control_comp._outputs['controls:{0}'.format(var)]
            else:
                val = control_comp._outputs[var]
                output = np.repeat(val, gd.num_nodes, axis=0)

        elif var_type == 'input_control':
            parameter_comp = self.input_controls
            if self.control_options[var]['dynamic']:
                output = parameter_comp._outputs['controls:{0}_out'.format(var)]
            else:
                val = parameter_comp._outputs['controls:{0}_out'.format(var)]
                output = np.repeat(val, gd.num_nodes, axis=0)

        elif var_type == 'control_rate':
            control_rate_comp = self.control_rate_comp
            output = control_rate_comp._outputs['control_rates:{0}_rate'.format(var)]

        elif var_type == 'parameter_rate':
            control_rate_comp = self.control_rate_comp
            output = control_rate_comp._outputs['parameter_rates:{0}_rate'.format(var)]

        elif var_type == 'rhs':
            raise NotImplementedError()

        return output[gd.subset_node_indices[nodes]]

    def _to_ozone_ode(self, ode_options):
        """
        Constructs an ODEFunction suitable for use in an Ozone integrator.

        Parameters
        ----------
        ode_options : ODEOptions
            Container for time, state, and parameter options.

        Returns
        -------
        new_ode_options : ODEOptions
            Modified container for time, state, and parameter options.
        """
        new_ode_options = ODEOptions()

        new_ode_options._time_options = ode_options._time_options
        new_ode_options._states = ode_options._states

        for control, opts in iteritems(self.control_options):
            if opts['dynamic']:
                new_ode_options._dynamic_parameters[control] = opts
                new_ode_options._dynamic_parameters[control]['targets'] = \
                    ode_options._dynamic_parameters[control]['targets']

        return new_ode_options

    def _setup_path_constraints(self):
        pass

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
            obj_path = 'times'
        elif var_type == 'state':
            obj_path = 'state:{0}'.format(name)
        elif var_type == 'indep_control':
            obj_path = 'dynamic_parameter:{0}'.format(name)
        elif var_type == 'input_control':
            obj_path = 'dynamic_parameter:{0}'.format(name)
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
