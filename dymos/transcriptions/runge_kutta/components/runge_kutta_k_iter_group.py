import openmdao.api as om

from ....utils.rk_methods import rk_methods
from ....utils.introspection import get_targets

from .runge_kutta_state_predict_comp import RungeKuttaStatePredictComp
from .runge_kutta_k_comp import RungeKuttaKComp


class RungeKuttaKIterGroup(om.Group):
    """
    This Group contains the state prediction component, the ODE, and the k-calculation component.
    Given the initial values of the states, the times at the ODE evaluations, and the stepsize
    across each segment it will iterate to find the Runge-Kutta weights 'k'.
    """
    def initialize(self):

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

        self.options.declare('solver_class', default=om.NonlinearBlockGS,
                             values=(om.NonlinearBlockGS, om.NewtonSolver, om.NonlinearRunOnce),
                             allow_none=True,
                             desc='The nonlinear solver class used to converge the numerical '
                                  'integration of the segment.')

        self.options.declare('solver_options', default={}, types=(dict,),
                             desc='The options passed to the nonlinear solver used to converge the'
                                  'Runge-Kutta propagation across each step.')

    def setup(self):
        num_seg = self.options['num_segments']
        rk_data = rk_methods[self.options['method']]
        num_nodes = num_seg * rk_data['num_stages']
        state_options = self.options['state_options']

        self.add_subsystem('state_predict_comp',
                           RungeKuttaStatePredictComp(method=self.options['method'],
                                                      num_segments=num_seg,
                                                      state_options=state_options))
        self.add_subsystem('ode',
                           subsys=self.options['ode_class'](num_nodes=num_nodes,
                                                            **self.options['ode_init_kwargs']))

        self.add_subsystem('k_comp',
                           subsys=RungeKuttaKComp(method=self.options['method'],
                                                  num_segments=num_seg,
                                                  state_options=state_options,
                                                  time_units=self.options['time_units']))

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        self.state_predict_comp.configure_io()
        self.k_comp.configure_io()

        for state_name, options in self.options['state_options'].items():
            targets = get_targets(ode=self.ode, name=state_name, user_targets=options['targets'])

            # Connect the state predicted (assumed) value to its targets in the ode
            if targets:
                self.connect('state_predict_comp.predicted_states:{0}'.format(state_name),
                             ['ode.{0}'.format(tgt) for tgt in targets])

            # Connect the k value associated with the state to the state predict comp
            self.connect('k_comp.k:{0}'.format(state_name),
                         'state_predict_comp.k:{0}'.format(state_name))

            # Delayed promotes
            self.promotes('state_predict_comp', inputs=[f'initial_states_per_seg:{state_name}'])
            self.promotes('k_comp', inputs=['h'])

        self.linear_solver = om.DirectSolver()
        if self.options['solver_class']:
            self.nonlinear_solver = self.options['solver_class'](**self.options['solver_options'])
