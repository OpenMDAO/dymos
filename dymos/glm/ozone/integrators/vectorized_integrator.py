import numpy as np
from six import iteritems

from openmdao.api import Group, IndepVarComp, LinearBlockGS, NonlinearBlockGS

from dymos.glm.ozone.integrators.integrator import Integrator
from dymos.glm.ozone.components.vectorized_step_comp import VectorizedStepComp
from dymos.glm.ozone.components.vectorized_stagestep_comp import VectorizedStageStepComp
from dymos.glm.ozone.components.vectorized_output_comp import VectorizedOutputComp
from dymos.glm.ozone.utils.var_names import get_name


class VectorizedIntegrator(Integrator):
    """
    Integrate an explicit method with a relaxed time-marching approach.
    """

    def initialize(self):
        super(VectorizedIntegrator, self).initialize()

        self.metadata.declare(
            'formulation', default='solver-based', values=['solver-based', 'optimizer-based'])

    def setup(self):
        super(VectorizedIntegrator, self).setup()

        ode_class = self.metadata['ode_class']
        ode_options = ode_class.ode_options
        method = self.metadata['method']
        starting_coeffs = self.metadata['starting_coeffs']
        formulation = self.metadata['formulation']

        has_starting_method = method.starting_method is not None
        is_starting_method = starting_coeffs is not None

        states = ode_options._states
        dynamic_parameters = ode_options._dynamic_parameters
        time_units = ode_options._time_options['units']

        starting_norm_times, my_norm_times = self._get_meta()

        glm_A, glm_B, glm_U, glm_V, num_stages, num_step_vars = self._get_method()

        num_times = len(my_norm_times)

        # ------------------------------------------------------------------------------------

        integration_group = Group()
        self.add_subsystem('integration_group', integration_group)

        if formulation == 'optimizer-based':
            comp = IndepVarComp()
            for state_name, state in iteritems(states):
                comp.add_output(
                    'Y:%s' % state_name,
                    shape=(num_times - 1, num_stages,) + state['shape'],
                    units=state['units'])
                kwargs = {
                    name: self.metadata['state_options'][state_name][name]
                    for name in ['lower', 'upper', 'scaler', 'adder', 'ref', 'ref0']
                }
                comp.add_design_var('Y:%s' % state_name, **kwargs)
            integration_group.add_subsystem('desvars_comp', comp)
        elif formulation == 'solver-based':
            comp = IndepVarComp()
            for state_name, state in iteritems(states):
                comp.add_output(
                    'Y:%s' % state_name, val=0.,
                    shape=(num_times - 1, num_stages,) + state['shape'],
                    units=state['units'])
            integration_group.add_subsystem('dummy_comp', comp)

        comp = self._create_ode((num_times - 1) * num_stages)
        integration_group.add_subsystem('ode_comp', comp)
        if ode_options._time_options['targets']:
            self.connect(
                'time_comp.stage_times',
                [
                    '.'.join(('integration_group.ode_comp', t))
                    for t in ode_options._time_options['targets']
                ],
            )
        if len(dynamic_parameters) > 0:
            self._connect_multiple(
                self._get_dynamic_parameter_names('dynamic_parameter_comp', 'out'),
                self._get_dynamic_parameter_names('integration_group.ode_comp', 'targets'),
            )

        comp = VectorizedStageStepComp(
            states=states, time_units=time_units,
            num_times=num_times, num_stages=num_stages, num_step_vars=num_step_vars,
            glm_A=glm_A, glm_U=glm_U, glm_B=glm_B, glm_V=glm_V,
        )
        integration_group.add_subsystem('vectorized_stagestep_comp', comp)
        self.connect('time_comp.h_vec', 'integration_group.vectorized_stagestep_comp.h_vec')

        comp = VectorizedStepComp(
            states=states, time_units=time_units,
            num_times=num_times, num_stages=num_stages, num_step_vars=num_step_vars,
            glm_B=glm_B, glm_V=glm_V,
        )
        self.add_subsystem('vectorized_step_comp', comp)
        self.connect('time_comp.h_vec', 'vectorized_step_comp.h_vec')
        self._connect_multiple(
            self._get_state_names('starting_system', 'starting'),
            self._get_state_names('integration_group.vectorized_stagestep_comp', 'y0'),
        )
        self._connect_multiple(
            self._get_state_names('starting_system', 'starting'),
            self._get_state_names('vectorized_step_comp', 'y0'),
        )

        comp = VectorizedOutputComp(
            states=states,
            num_starting_times=len(starting_norm_times), num_my_times=len(my_norm_times),
            num_step_vars=num_step_vars, starting_coeffs=starting_coeffs,
        )

        promotes = []
        promotes.extend([get_name('state', state_name) for state_name in states])
        if is_starting_method:
            promotes.extend([get_name('starting', state_name) for state_name in states])

        self.add_subsystem('output_comp', comp, promotes_outputs=promotes)
        if has_starting_method:
            self._connect_multiple(
                self._get_state_names('starting_system', 'state'),
                self._get_state_names('output_comp', 'starting_state'),
            )

        src_indices_to_ode = []
        src_indices_from_ode = []
        for state_name, state in iteritems(states):
            size = np.prod(state['shape'])
            shape = state['shape']

            src_indices_to_ode.append(
                np.arange((num_times - 1) * num_stages * size).reshape(
                    ((num_times - 1) * num_stages,) + shape))

            src_indices_from_ode.append(
                np.arange((num_times - 1) * num_stages * size).reshape(
                    (num_times - 1, num_stages,) + shape))

        src_indices_to_ode = [np.array(idx).squeeze() for idx in src_indices_to_ode]

        self._connect_multiple(
            self._get_state_names('vectorized_step_comp', 'y'),
            self._get_state_names('output_comp', 'y'),
        )

        self._connect_multiple(
            self._get_state_names('integration_group.ode_comp', 'rate_source'),
            self._get_state_names('vectorized_step_comp', 'F'),
            src_indices_from_ode,
        )
        self._connect_multiple(
            self._get_state_names('integration_group.ode_comp', 'rate_source'),
            self._get_state_names('integration_group.vectorized_stagestep_comp', 'F'),
            src_indices_from_ode,
        )

        if formulation == 'solver-based':
            self._connect_multiple(
                self._get_state_names('integration_group.vectorized_stagestep_comp', 'Y_out'),
                self._get_state_names('integration_group.ode_comp', 'targets'),
                src_indices_to_ode,
            )
            self._connect_multiple(
                self._get_state_names('integration_group.dummy_comp', 'Y'),
                self._get_state_names('integration_group.vectorized_stagestep_comp', 'Y_in'),
            )
        elif formulation == 'optimizer-based':
            self._connect_multiple(
                self._get_state_names('integration_group.desvars_comp', 'Y'),
                self._get_state_names('integration_group.ode_comp', 'targets'),
                src_indices_to_ode,
            )
            self._connect_multiple(
                self._get_state_names('integration_group.desvars_comp', 'Y'),
                self._get_state_names('integration_group.vectorized_stagestep_comp', 'Y_in'),
            )
            for state_name, state in iteritems(states):
                defect_scaler = self.metadata['state_options'][state_name]['defect_scaler']
                integration_group.add_constraint(
                    'vectorized_stagestep_comp.Y_out:%s' % state_name,
                    equals=0., vectorize_derivs=True, scaler=defect_scaler,
                )

        if has_starting_method:
            self.starting_system.metadata['formulation'] = self.metadata['formulation']

        if formulation == 'solver-based':
            integration_group.nonlinear_solver = NonlinearBlockGS(
                iprint=2, maxiter=40, atol=1e-14, rtol=1e-12)

            integration_group.linear_solver = LinearBlockGS(
                iprint=1, maxiter=40, atol=1e-14, rtol=1e-12)
