import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp

from dymos.glm.ozone.integrators.integrator import Integrator
from dymos.glm.ozone.components.starting_comp import StartingComp
from dymos.glm.ozone.components.explicit_tm_stage_comp import ExplicitTMStageComp
from dymos.glm.ozone.components.explicit_tm_step_comp import ExplicitTMStepComp
from dymos.glm.ozone.components.tm_output_comp import TMOutputComp
from dymos.glm.ozone.utils.var_names import get_name


class ExplicitTMIntegrator(Integrator):
    """
    Integrate an explicit method with a time-marching approach.
    """

    def setup(self):
        super(ExplicitTMIntegrator, self).setup()

        ode_class = self.metadata['ode_class']
        method = self.metadata['method']
        starting_coeffs = self.metadata['starting_coeffs']

        has_starting_method = method.starting_method is not None
        is_starting_method = starting_coeffs is not None

        states = ode_class.ode_options._states
        dynamic_parameters = ode_class.ode_options._dynamic_parameters
        time_units = ode_class.ode_options._time_options['units']

        starting_norm_times, my_norm_times = self._get_meta()

        glm_A, glm_B, glm_U, glm_V, num_stages, num_step_vars = self._get_method()

        num_times = len(my_norm_times)
        num_stages = method.num_stages
        num_step_vars = method.num_values

        glm_A = method.A
        glm_B = method.B
        glm_U = method.U
        glm_V = method.V

        # ------------------------------------------------------------------------------------

        integration_group = Group()
        self.add_subsystem('integration_group', integration_group)

        for i_step in range(len(my_norm_times) - 1):

            step_comp_old_name = 'integration_group.step_comp_%i' % (i_step - 1)
            step_comp_new_name = 'integration_group.step_comp_%i' % (i_step)

            for i_stage in range(num_stages):
                stage_comp_name = 'integration_group.stage_comp_%i_%i' % (i_step, i_stage)
                ode_comp_name = 'integration_group.ode_comp_%i_%i' % (i_step, i_stage)

                if ode_class.ode_options._time_options['targets']:
                    self.connect('time_comp.stage_times',
                        ['.'.join((ode_comp_name, t)) for t in
                         ode_class.ode_options._time_options['targets']],
                        src_indices=i_step * (num_stages) + i_stage
                    )

                comp = ExplicitTMStageComp(
                    states=states, time_units=time_units,
                    num_stages=num_stages, num_step_vars=num_step_vars,
                    glm_A=glm_A, glm_U=glm_U, i_stage=i_stage, i_step=i_step,
                )
                integration_group.add_subsystem(stage_comp_name.split('.')[1], comp)
                self.connect('time_comp.h_vec', '%s.h' % stage_comp_name, src_indices=i_step)

                for j_stage in range(i_stage):
                    ode_comp_tmp_name = 'integration_group.ode_comp_%i_%i' % (i_step, j_stage)
                    self._connect_multiple(
                        self._get_state_names(ode_comp_tmp_name, 'rate_source'),
                        self._get_state_names(stage_comp_name, 'F', i_step=i_step, i_stage=i_stage, j_stage=j_stage),
                    )

                comp = self._create_ode(1)
                integration_group.add_subsystem(ode_comp_name.split('.')[1], comp)
                self._connect_multiple(
                    self._get_state_names(stage_comp_name, 'Y', i_step=i_step, i_stage=i_stage),
                    self._get_state_names(ode_comp_name, 'targets'),
                )

                if len(dynamic_parameters) > 0:
                    src_indices_list = []
                    for parameter_name, value in iteritems(dynamic_parameters):
                        size = np.prod(value['shape'])
                        shape = value['shape']

                        arange = np.arange(((len(my_norm_times) - 1) * num_stages * size)).reshape(
                            ((len(my_norm_times) - 1, num_stages,) + shape))
                        src_indices = arange[i_step, i_stage, :]
                        src_indices_list.append(src_indices)
                    self._connect_multiple(
                        self._get_dynamic_parameter_names('dynamic_parameter_comp', 'out'),
                        self._get_dynamic_parameter_names(ode_comp_name, 'targets'),
                        src_indices_list,
                    )

            comp = ExplicitTMStepComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_B=glm_B, glm_V=glm_V, i_step=i_step,
            )
            integration_group.add_subsystem(step_comp_new_name.split('.')[1], comp)
            self.connect('time_comp.h_vec', '%s.h' % step_comp_new_name, src_indices=i_step)
            for j_stage in range(num_stages):
                ode_comp_tmp_name = 'integration_group.ode_comp_%i_%i' % (i_step, j_stage)
                self._connect_multiple(
                    self._get_state_names(ode_comp_tmp_name, 'rate_source'),
                    self._get_state_names(step_comp_new_name, 'F', i_step=i_step, j_stage=j_stage),
                )

            if i_step == 0:
                self._connect_multiple(
                    self._get_state_names('starting_system', 'starting'),
                    self._get_state_names(step_comp_new_name, 'y_old', i_step=i_step),
                )
                for i_stage in range(num_stages):
                    stage_comp_name = 'integration_group.stage_comp_%i_%i' % (i_step, i_stage)
                    self._connect_multiple(
                        self._get_state_names('starting_system', 'starting'),
                        self._get_state_names(stage_comp_name, 'y_old', i_step=i_step, i_stage=i_stage),
                    )
            else:
                self._connect_multiple(
                    self._get_state_names(step_comp_old_name, 'y_new', i_step=i_step - 1),
                    self._get_state_names(step_comp_new_name, 'y_old', i_step=i_step),
                )
                for i_stage in range(num_stages):
                    stage_comp_name = 'integration_group.stage_comp_%i_%i' % (i_step, i_stage)
                    self._connect_multiple(
                        self._get_state_names(step_comp_old_name, 'y_new', i_step=i_step - 1),
                        self._get_state_names(stage_comp_name, 'y_old', i_step=i_step, i_stage=i_stage),
                    )

        promotes_outputs = []
        for state_name in states:
            out_state_name = get_name('state', state_name)
            starting_name = get_name('starting', state_name)
            promotes_outputs.append(out_state_name)
            if is_starting_method:
                promotes_outputs.append(starting_name)

        comp = TMOutputComp(
            states=states, num_starting_times=len(starting_norm_times),
            num_my_times=len(my_norm_times), num_step_vars=num_step_vars,
            starting_coeffs=starting_coeffs)
        self.add_subsystem('output_comp', comp, promotes_outputs=promotes_outputs)
        if has_starting_method:
            self._connect_multiple(
                self._get_state_names('starting_system', 'state'),
                self._get_state_names('output_comp', 'starting_state'),
            )

        for i_step in range(len(my_norm_times)):
            if i_step == 0:
                self._connect_multiple(
                    self._get_state_names('starting_system', 'starting'),
                    self._get_state_names('output_comp', 'y', i_step=i_step),
                )
            else:
                self._connect_multiple(
                    self._get_state_names(
                        'integration_group.step_comp_%i' % (i_step - 1), 'y_new', i_step=i_step - 1),
                    self._get_state_names('output_comp', 'y', i_step=i_step),
                )
