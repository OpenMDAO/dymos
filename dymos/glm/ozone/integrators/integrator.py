from __future__ import division, print_function

import numpy as np
from openmdao.api import Group, IndepVarComp
from openmdao.core.system import System
from six import iteritems

import dymos.glm.ozone.methods.method as methods
from dymos.glm.ozone.components.initial_conditions_comp import InitialConditionsComp
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
        self.metadata.declare('ode_class')
        self.metadata.declare('ode_init_kwargs', types=dict, allow_none=True, default={})
        self.metadata.declare('method', types=GLMMethod)
        self.metadata.declare('starting_coeffs', types=np.ndarray, allow_none=True, default=None)

        self.metadata.declare('initial_conditions', types=dict, allow_none=True, default=None)
        self.metadata.declare('dynamic_parameters', types=dict, allow_none=True, default=None)

        self.metadata.declare('initial_time', default=None)
        self.metadata.declare('final_time', default=None)
        self.metadata.declare('normalized_times', types=np.ndarray)
        self.metadata.declare('all_norm_times', types=np.ndarray)

        self.metadata.declare('state_options', types=dict)

    def setup(self):
        ode_class = self.metadata['ode_class']
        self._ode_options = ode_class.ode_options
        method = self.metadata['method']
        starting_coeffs = self.metadata['starting_coeffs']

        initial_conditions = self.metadata['initial_conditions']
        given_dynamic_parameters = self.metadata['dynamic_parameters']

        initial_time = self.metadata['initial_time']
        final_time = self.metadata['final_time']

        num_step_vars = method.num_values

        states = self._ode_options._states
        dynamic_parameters = self._ode_options._dynamic_parameters
        time_units = self._ode_options._time_options['units']

        starting_norm_times, my_norm_times = self._get_meta()
        stage_norm_times = self._get_stage_norm_times()
        all_norm_times = self.metadata['all_norm_times']
        normalized_times = self.metadata['normalized_times']

        # ------------------------------------------------------------------------------------
        # inputs
        comp = IndepVarComp()
        promotes = []

        # # Dummy state history to pull out initial conditions to conform to pointer's api
        # for state_name, state in iteritems(states):
        #     name = 'IC_state:%s' % state_name
        #     state = self.metadata['state_options'][state_name]
        #
        #     comp.add_output(
        #         name,
        #         shape=(len(normalized_times),) + state['shape'], units=state['units']
        #     )
        #     promotes.append(name)
        #
        #     if not state['fix_initial']:
        #         lower = state['lower']
        #         upper = state['upper']
        #         if lower is not None and not np.isscalar(lower):
        #             lower = lower[0]
        #         if upper is not None and not np.isscalar(upper):
        #             upper = upper[0]
        #
        #         comp.add_design_var(
        #             name, indices=np.arange(np.prod(state['shape'])),
        #             lower=lower, upper=upper,
        #         )

        # Initial conditions
        for state_name, state in iteritems(states):
            name = get_name('initial_condition', state_name)
            state = self.metadata['state_options'][state_name]

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

        # # ------------------------------------------------------------------------------------
        # # Initial conditions comp
        # comp = InitialConditionsComp(states=states)
        # self.add_subsystem(
        #     'initial_conditions_comp', comp, promotes_outputs=[
        #         ('out:%s' % state_name, get_name('initial_condition', state_name))
        #         for state_name in states
        #     ])
        # for state_name in states:
        #     size = np.prod(state['shape'])
        #     self.connect(
        #         'IC_state:%s' % state_name, 'initial_conditions_comp.in:%s' % state_name,
        #         src_indices=np.arange(size), flat_src_indices=True)

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
        ode_class = self.metadata['ode_class']
        init_kwargs = self.metadata['ode_init_kwargs']
        init_kwargs = init_kwargs if init_kwargs is not None else {}
        return ode_class(num_nodes=num, **init_kwargs)

    def _get_meta(self):
        method = self.metadata['method']
        normalized_times = self.metadata['normalized_times']

        start_time_index = 0

        return normalized_times[:start_time_index + 1], normalized_times[start_time_index:]

    def _get_method(self):
        method = self.metadata['method']

        return method.A, method.B, method.U, method.V, method.num_stages, method.num_values

    def _get_stage_norm_times(self):
        starting_norm_times, my_norm_times = self._get_meta()

        abscissa = self.metadata['method'].abscissa

        repeated_times1 = np.repeat(my_norm_times[:-1], len(abscissa))
        repeated_times2 = np.repeat(my_norm_times[1:], len(abscissa))
        tiled_abscissa = np.tile(abscissa, len(my_norm_times) - 1)

        stage_norm_times = repeated_times1 + (repeated_times2 - repeated_times1) * tiled_abscissa

        return stage_norm_times
