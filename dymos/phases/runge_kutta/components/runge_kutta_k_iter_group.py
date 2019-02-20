from __future__ import print_function, division, absolute_import

from six import string_types, iteritems

from openmdao.api import Group, DirectSolver, NonlinearBlockGS, NewtonSolver, NonlinearRunOnce

from ....utils.rk_methods import rk_methods

from .runge_kutta_state_predict_comp import RungeKuttaStatePredictComp
from .runge_kutta_k_comp import RungeKuttaKComp


class RungeKuttaKIterGroup(Group):
    """
    This Group contains the state prediction component, the ODE, and the k-calculation component.
    Given the initial values of the states, the times at the ODE evaluations, and the stepsize
    across each segment it will iterate to find the Runge-Kutta weights 'k'.
    """
    def __init__(self, get_rate_source_path, **kwargs):
        """

        Parameters
        ----------
        get_rate_source_path : callable
            The function or method used to provide state rate source path information.  Nominally
            this is the _get_rate_source_path method of the parent phase but for testing purposes
            it is convenient to override it so an entire Phase does not need to be included in the
            test.
        """
        self._get_rate_source_path = get_rate_source_path
        super(RungeKuttaKIterGroup, self).__init__(**kwargs)

    def initialize(self):

        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase')

        self.options.declare('method', default='rk4', values=('rk4',),
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of the integration variable')

        self.options.declare('ode_class',
                             desc='System defining the ODE')

        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')

        self.options.declare('solver_class', default=NonlinearBlockGS,
                             values=(NonlinearBlockGS, NewtonSolver, NonlinearRunOnce),
                             allow_none=True,
                             desc='The nonlinear solver class used to converge the numerical '
                                  'integration of the segment.')

        self.options.declare('solver_options', default={}, types=(dict,),
                             desc='The options passed to the nonlinear solver used to converge the'
                                  'Runge-Kutta propagation.')


    def setup(self):
        num_seg = self.options['num_segments']
        rk_data = rk_methods[self.options['method']]
        num_nodes = num_seg * rk_data['num_stages']
        state_options = self.options['state_options']

        self.add_subsystem('state_predict_comp',
                           RungeKuttaStatePredictComp(method=self.options['method'],
                                                      num_segments=num_seg,
                                                      state_options=state_options),
                           promotes_inputs=['initial_states:*'])
        self.add_subsystem('ode',
                           subsys=self.options['ode_class'](num_nodes=num_nodes,
                                                            **self.options['ode_init_kwargs']))

        self.add_subsystem('k_comp',
                           subsys=RungeKuttaKComp(method=self.options['method'],
                                                  num_segments=num_seg,
                                                  state_options=state_options,
                                                  time_units=self.options['time_units']),
                           promotes_inputs=['h'])

        for state_name, options in iteritems(self.options['state_options']):
            # Connect the state predicted (assumed) value to its targets in the ode
            self.connect('state_predict_comp.predicted_states:{0}'.format(state_name),
                             ['ode.{0}'.format(tgt) for tgt in options['targets']])

            # Connect the state rate source to the k comp
            rate_path, src_idxs = self._get_rate_source_path(state_name)
            self.connect(rate_path,
                         'k_comp.f:{0}'.format(state_name),
                         src_indices=src_idxs,
                         flat_src_indices=True)

            # Connect the k value associated with the state to the state predict comp
            self.connect('k_comp.k:{0}'.format(state_name),
                         'state_predict_comp.k:{0}'.format(state_name))

        self.linear_solver = DirectSolver()
        if self.options['solver_class']:
            self.nonlinear_solver = self.options['solver_class'](**self.options['solver_options'])
