from __future__ import division, print_function

import numpy as np
from openmdao.api import Group, IndepVarComp
from openmdao.core.system import System
from six import iteritems

import dymos.glm.ozone.methods.method as methods
from dymos.glm.ozone.components.time_comp import TimeComp
from dymos.glm.ozone.components.starting_comp import StartingComp
from dymos.glm.ozone.components.dynamic_parameter_comp import DynamicParameterComp
from dymos.glm.ozone.methods.method import GLMMethod
from dymos.glm.ozone.utils.var_names import get_name
from dymos.glm.ozone.methods_list import get_method


class Integrator(Group):
    """
    The base class for all integrators.
    """

    def initialize(self):
        self.options.declare('ode_class')
        self.options.declare('ode_init_kwargs', types=dict, allow_none=True, default={})
        self.options.declare('method', types=GLMMethod)
        self.options.declare('starting_coeffs', types=np.ndarray, allow_none=True, default=None)

        self.options.declare('initial_conditions', types=dict, allow_none=True, default=None)
        self.options.declare('dynamic_parameters', types=dict, allow_none=True, default=None)

        self.options.declare('initial_time', default=None)
        self.options.declare('final_time', default=None)
        self.options.declare('normalized_times', types=np.ndarray)
        self.options.declare('all_norm_times', types=np.ndarray)

        self.options.declare('state_options', types=dict)

    def setup(self):
        ode_class = self.options['ode_class']
        self._ode_options = ode_class.ode_options
        method = self.options['method']
        starting_coeffs = self.options['starting_coeffs']

        initial_conditions = self.options['initial_conditions']
        given_dynamic_parameters = self.options['dynamic_parameters']

        initial_time = self.options['initial_time']
        final_time = self.options['final_time']

        num_step_vars = method.num_values

        states = self._ode_options._states
        dynamic_parameters = self._ode_options._dynamic_parameters
        time_units = self._ode_options._time_options['units']

        starting_norm_times, my_norm_times = self._get_meta()
        stage_norm_times = self._get_stage_norm_times()
        all_norm_times = self.options['all_norm_times']
        normalized_times = self.options['normalized_times']

        # ------------------------------------------------------------------------------------
        # inputs
        comp = IndepVarComp()
        promotes = []

        # Initial conditions
        for state_name, state in iteritems(states):
            name = get_name('initial_condition', state_name)
            state = self.options['state_options'][state_name]

            comp.add_output(name, units=state['units'])
            promotes.append(name)

            if not state['fix_initial']:
                lower = state['lower']
                upper = state['upper']
                if lower is not None and not np.isscalar(lower):
                    lower = lower[0]
                if upper is not None and not np.isscalar(upper):
                    upper = upper[0]
                scaler = state['scaler']

                comp.add_design_var(name, lower=lower, upper=upper, scaler=scaler)

        # Given dynamic_parameters
        if given_dynamic_parameters is not None:
            for parameter_name, value in iteritems(given_dynamic_parameters):
                name = get_name('dynamic_parameter', parameter_name)
                parameter = self._ode_options._dynamic_parameters[parameter_name]

                comp.add_output(name, val=value, units=parameter['units'])
                promotes.append(name)

        # Initial time
        if initial_time is not None:
            comp.add_output('initial_time', val=initial_time, units=time_units)
            promotes.append('initial_time')

        # Final time
        if final_time is not None:
            comp.add_output('final_time', val=final_time, units=time_units)
            promotes.append('final_time')

        self.add_subsystem('inputs', comp, promotes_outputs=promotes)

        # ------------------------------------------------------------------------------------
        # Time comp
        comp = TimeComp(
            time_units=time_units, my_norm_times=my_norm_times, stage_norm_times=stage_norm_times,
            normalized_times=normalized_times)
        self.add_subsystem(
            'time_comp', comp,
            promotes_inputs=['initial_time', 'final_time'],
            promotes_outputs=['times'])

        # ------------------------------------------------------------------------------------
        # Dynamic parameter comp
        if len(dynamic_parameters) > 0:
            promotes = [
                (get_name('in', parameter_name), get_name('dynamic_parameter', parameter_name))
                for parameter_name in dynamic_parameters]
            self.add_subsystem(
                'dynamic_parameter_comp',
                DynamicParameterComp(
                    dynamic_parameters=dynamic_parameters,
                    normalized_times=all_norm_times, stage_norm_times=stage_norm_times),
                promotes_inputs=promotes)

        # ------------------------------------------------------------------------------------
        # Starting system
        promotes = []
        promotes.extend([get_name('initial_condition', state_name) for state_name in states])

        starting_system = StartingComp(states=states, num_step_vars=num_step_vars)

        self.add_subsystem('starting_system', starting_system, promotes_inputs=promotes)

    def _get_state_names(self, comp, type_, i_step=None, i_stage=None, j_stage=None):
        return self._get_names(
            'states', comp, type_, i_step=i_step, i_stage=i_stage, j_stage=j_stage)

    def _get_dynamic_parameter_names(self, comp, type_, i_step=None, i_stage=None, j_stage=None):
        return self._get_names(
            'dynamic_parameters', comp, type_, i_step=i_step, i_stage=i_stage, j_stage=j_stage)

    def _get_names(self, variable_type, comp, type_, i_step=None, i_stage=None, j_stage=None):
        if variable_type == 'states':
            variables_dict = self._ode_options._states
        elif variable_type == 'dynamic_parameters':
            variables_dict = self._ode_options._dynamic_parameters

        names_list = []
        for variable_name, variable in iteritems(variables_dict):
            if type_ == 'rate_source':
                names = '{}.{}'.format(comp, variable['rate_source'])
            elif type_ == 'targets':
                names = ['{}.{}'.format(comp, tgt) for tgt in variable['targets']] \
                    if variable['targets'] else []
            else:
                names = '{}.{}'.format(comp, get_name(
                    type_, variable_name, i_step=i_step, i_stage=i_stage, j_stage=j_stage))

            names_list.append(names)

        return names_list

    def _connect_multiple(self, srcs_list, tgts_list, src_indices_list=None):
        if src_indices_list is None:
            for srcs, tgts in zip(srcs_list, tgts_list):
                self.connect(srcs, tgts)
        else:
            for srcs, tgts, src_indices in zip(srcs_list, tgts_list, src_indices_list):
                self.connect(srcs, tgts, src_indices=src_indices, flat_src_indices=True)

    def _create_ode(self, num):
        ode_class = self.options['ode_class']
        init_kwargs = self.options['ode_init_kwargs']
        init_kwargs = init_kwargs if init_kwargs is not None else {}
        return ode_class(num_nodes=num, **init_kwargs)

    def _get_meta(self):
        method = self.options['method']
        normalized_times = self.options['normalized_times']

        start_time_index = 0

        return normalized_times[:start_time_index + 1], normalized_times[start_time_index:]

    def _get_method(self):
        method = self.options['method']

        return method.A, method.B, method.U, method.V, method.num_stages, method.num_values

    def _get_stage_norm_times(self):
        starting_norm_times, my_norm_times = self._get_meta()

        abscissa = self.options['method'].abscissa

        repeated_times1 = np.repeat(my_norm_times[:-1], len(abscissa))
        repeated_times2 = np.repeat(my_norm_times[1:], len(abscissa))
        tiled_abscissa = np.tile(abscissa, len(my_norm_times) - 1)

        stage_norm_times = repeated_times1 + (repeated_times2 - repeated_times1) * tiled_abscissa

        return stage_norm_times
