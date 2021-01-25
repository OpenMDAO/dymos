import numpy as np
import openmdao.api as om

from .runge_kutta_k_iter_group import RungeKuttaKIterGroup
from .runge_kutta_state_advance_comp import RungeKuttaStateAdvanceComp
from .runge_kutta_state_continuity_comp import RungeKuttaStateContinuityComp
from ....utils.indexing import get_src_indices_by_row


class RungeKuttaStateContinuityIterGroup(om.Group):
    """
    Class definition for the RungeKuttaStateContinuityIterGroup.

    This Group contains the k-iteration subgroup, the state advance component, continuity defect
    component.  Given the initial value of each state at the beginning of the first segment,
    the times at the ODE evaluations, and the stepsize across each segment it will iterate
    to find the Runge-Kutta weights 'k' and the initial values of the states in all but the first
    segments to provide state continuity across the phase.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase')

        self.options.declare('method', default='RK4', types=str,
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of the integration variable')

        self.options.declare('ode_class',
                             desc='System defining the ODE')

        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')

        self.options.declare('k_solver_class', default=om.NonlinearBlockGS,
                             values=(om.NonlinearBlockGS, om.NewtonSolver, om.NonlinearRunOnce),
                             allow_none=True,
                             desc='The nonlinear solver class used to converge the numerical '
                                  'integration across each segment.')

        self.options.declare('k_solver_options', default={'iprint': -1}, types=(dict,),
                             desc='The options passed to the nonlinear solver used to converge the'
                                  'Runge-Kutta propagation across each step.')

    def setup(self):
        """
        Build the group hierarchy.
        """
        self.add_subsystem('k_iter_group',
                           RungeKuttaKIterGroup(num_segments=self.options['num_segments'],
                                                method=self.options['method'],
                                                state_options=self.options['state_options'],
                                                time_units=self.options['time_units'],
                                                ode_class=self.options['ode_class'],
                                                ode_init_kwargs=self.options['ode_init_kwargs'],
                                                solver_class=self.options['k_solver_class'],
                                                solver_options=self.options['k_solver_options']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        self.add_subsystem('state_advance_comp',
                           RungeKuttaStateAdvanceComp(num_segments=self.options['num_segments'],
                                                      method=self.options['method'],
                                                      state_options=self.options['state_options']))

        self.add_subsystem('continuity_comp',
                           RungeKuttaStateContinuityComp(
                               num_segments=self.options['num_segments'],
                               state_options=self.options['state_options']))

        self.linear_solver = om.DirectSolver()

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        self.k_iter_group.configure_io()
        self.state_advance_comp.configure_io()
        self.continuity_comp.configure_io()

        num_connected = len([state for state in self.options['state_options'] if
                             self.options['state_options'][state]['connected_initial']])
        for state_name, options in self.options['state_options'].items():
            self.connect('k_comp.k:{0}'.format(state_name),
                         'state_advance_comp.k:{0}'.format(state_name))
            row_idxs = np.arange(self.options['num_segments'], dtype=int)
            src_idxs = get_src_indices_by_row(row_idxs, options['shape'])
            self.connect('states:{0}'.format(state_name),
                         'initial_states_per_seg:{0}'.format(state_name),
                         src_indices=src_idxs, flat_src_indices=True)

            # Delayed promotes
            self.promotes('state_advance_comp',
                          inputs=[f'initial_states_per_seg:{state_name}'],
                          outputs=[f'state_integrals:{state_name}', f'final_states:{state_name}'])

            if num_connected > 0:
                cc_promoted_inputs = [f'state_integrals:{state_name}', f'initial_states:{state_name}']
            else:
                cc_promoted_inputs = [f'state_integrals:{state_name}']

            self.promotes('continuity_comp',
                          inputs=cc_promoted_inputs,
                          outputs=[f'states:{state_name}'])
