import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, NewtonSolver, DirectSolver

from dymos.glm.ozone.integrators.integrator import Integrator
from dymos.glm.ozone.components.starting_comp import StartingComp
from dymos.glm.ozone.components.implicit_tm_stage_comp import ImplicitTMStageComp
from dymos.glm.ozone.components.implicit_tm_step_comp import ImplicitTMStepComp
from dymos.glm.ozone.components.tm_output_comp import TMOutputComp
from dymos.glm.ozone.utils.var_names import get_name


class ImplicitTMIntegrator(Integrator):
    """
    Integrate an implicit method with a time-marching approach.
    """

    def setup(self):
        super(ImplicitTMIntegrator, self).setup()

        ode_class = self.options['ode_class']
        method = self.options['method']
        starting_coeffs = self.options['starting_coeffs']

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
            group = Group()
            group_old_name = 'integration_group.step_%i' % (i_step - 1)
            group_new_name = 'integration_group.step_%i' % i_step
            integration_group.add_subsystem(group_new_name.split('.')[1], group)

            comp = self._create_ode(num_stages)
            group.add_subsystem('ode_comp', comp)
            if ode_class.ode_options._time_options['targets']:
                self.connect(
                    'time_comp.stage_times',
                    ['.'.join((group_new_name + '.ode_comp', t)) for t in
                     ode_class.ode_options._time_options['targets']],
                    src_indices=i_step * (num_stages) + np.arange(num_stages))

            if len(dynamic_parameters) > 0:
                src_indices_list = []
                for parameter_name, value in iteritems(dynamic_parameters):
                    size = np.prod(value['shape'])
                    shape = value['shape']

                    arange = np.arange(((len(my_norm_times) - 1) * num_stages * size)).reshape(
                        ((len(my_norm_times) - 1, num_stages,) + shape))
                    src_indices = arange[i_step, :, :]
                    src_indices_list.append(src_indices.flat)
                self._connect_multiple(
                    self._get_dynamic_parameter_names('dynamic_parameter_comp', 'out'),
                    self._get_dynamic_parameter_names(group_new_name + '.ode_comp', 'targets'),
                    src_indices_list,
                )

            comp = ImplicitTMStageComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_A=glm_A, glm_U=glm_U, i_step=i_step,
            )
            group.add_subsystem('stage_comp', comp)
            self.connect('time_comp.h_vec', group_new_name + '.stage_comp.h', src_indices=i_step)

            comp = ImplicitTMStepComp(
                states=states, time_units=time_units,
                num_stages=num_stages, num_step_vars=num_step_vars,
                glm_B=glm_B, glm_V=glm_V, i_step=i_step,
            )
            group.add_subsystem('step_comp', comp)
            self.connect('time_comp.h_vec', group_new_name + '.step_comp.h', src_indices=i_step)

            self._connect_multiple(
                self._get_state_names(group_new_name + '.ode_comp', 'rate_source'),
                self._get_state_names(group_new_name + '.step_comp', 'F', i_step=i_step),
            )

            self._connect_multiple(
                self._get_state_names(group_new_name + '.ode_comp', 'rate_source'),
                self._get_state_names(group_new_name + '.stage_comp', 'F', i_step=i_step),
            )

            self._connect_multiple(
                self._get_state_names(group_new_name + '.stage_comp', 'Y', i_step=i_step),
                self._get_state_names(group_new_name + '.ode_comp', 'targets'),
            )

            if i_step == 0:
                self._connect_multiple(
                    self._get_state_names('starting_system', 'starting'),
                    self._get_state_names(group_new_name + '.step_comp', 'y_old', i_step=i_step),
                )
                self._connect_multiple(
                    self._get_state_names('starting_system', 'starting'),
                    self._get_state_names(group_new_name + '.stage_comp', 'y_old', i_step=i_step),
                )
            else:
                self._connect_multiple(
                    self._get_state_names(
                        group_old_name + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_state_names(group_new_name + '.step_comp', 'y_old', i_step=i_step),
                )
                self._connect_multiple(
                    self._get_state_names(
                        group_old_name + '.step_comp', 'y_new', i_step=i_step - 1),
                    self._get_state_names(group_new_name + '.stage_comp', 'y_old', i_step=i_step),
                )

            group.nonlinear_solver = NewtonSolver(iprint=2, maxiter=100)
            group.linear_solver = DirectSolver(assemble_jac=True)
            group.options['assembled_jac_type'] = 'dense'

        promotes = []
        promotes.extend([get_name('state', state_name) for state_name in states])
        if is_starting_method:
            promotes.extend([get_name('starting', state_name) for state_name in states])

        comp = TMOutputComp(
            states=states, num_starting_times=len(starting_norm_times),
            num_my_times=len(my_norm_times), num_step_vars=num_step_vars,
            starting_coeffs=starting_coeffs)
        self.add_subsystem('output_comp', comp, promotes_outputs=promotes)
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
                        'integration_group.step_%i' % (i_step - 1) + '.step_comp',
                        'y_new', i_step=i_step - 1),
                    self._get_state_names('output_comp', 'y', i_step=i_step),
                )
