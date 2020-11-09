import numpy as np
from scipy.linalg import block_diag

import openmdao.api as om

from ....utils.rk_methods import rk_methods
from ....options import options as dymos_options


class RungeKuttaStateAdvanceComp(om.ExplicitComponent):
    """
    Given the initial value of each state at the start of each segment and the weight factors k
    for each state in each segment, compute the final value of each state at the end of each
    segment.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        self.options.declare('num_segments', types=int,
                             desc='The number of segments (time steps) in the phase')

        self.options.declare('method', default='RK4', types=str,
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        self._var_names = {}

        num_segs = self.options['num_segments']
        rk_data = rk_methods[self.options['method']]
        num_stages = rk_data['num_stages']

        for name, options in self.options['state_options'].items():
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            self._var_names[name] = {}
            self._var_names[name]['initial'] = 'initial_states_per_seg:{0}'.format(name)
            self._var_names[name]['k'] = 'k:{0}'.format(name)
            self._var_names[name]['final'] = 'final_states:{0}'.format(name)
            self._var_names[name]['integral'] = 'state_integrals:{0}'.format(name)

            self.add_input(self._var_names[name]['initial'], shape=(num_segs,) + shape, units=units,
                           desc='The initial value of the state at the start of each segment.')

            self.add_input(self._var_names[name]['k'], shape=(num_segs, num_stages) + shape,
                           units=units, desc='RK multiplier k for each stage in each segment.')

            self.add_output(self._var_names[name]['integral'], shape=(num_segs,) + shape,
                            units=units,
                            desc='The change in value of the state along each segment')

            self.add_output(self._var_names[name]['final'], shape=(num_segs,) + shape,
                            units=units,
                            desc='The final value of the state at the end of each segment.')

            e = np.eye(size*num_segs)
            r, c = np.nonzero(e)
            self.declare_partials(of=self._var_names[name]['final'],
                                  wrt=self._var_names[name]['initial'],
                                  rows=r, cols=c, val=1.0)

            p = np.kron(rk_data['b'], np.eye(size))
            p = block_diag(*num_segs*[p])
            r, c = p.nonzero()

            self.declare_partials(of=self._var_names[name]['final'],
                                  wrt=self._var_names[name]['k'],
                                  rows=r, cols=c,
                                  val=np.tile(rk_data['b'], size*num_segs))

            self.declare_partials(of=self._var_names[name]['integral'],
                                  wrt=self._var_names[name]['k'],
                                  rows=r, cols=c,
                                  val=np.tile(rk_data['b'], size*num_segs))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for name, options in self.options['state_options'].items():
            x0 = inputs[self._var_names[name]['initial']]
            k = inputs[self._var_names[name]['k']]
            b = rk_methods[self.options['method']]['b']
            final_name = self._var_names[name]['final']
            integral_name = self._var_names[name]['integral']
            outputs[integral_name] = np.einsum('ijk...,j...->ik...', k, b)
            outputs[final_name] = x0 + outputs[integral_name]
