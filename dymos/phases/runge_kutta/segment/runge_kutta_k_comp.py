from __future__ import print_function, division, absolute_import

from six import string_types, iteritems

import numpy as np

from openmdao.api import ExplicitComponent

from ....utils.rk_methods import rk_methods
from ....utils.misc import get_rate_units


class RungeKuttaKComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('method', default='rk4', values=('rk4',),
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of the integration variable')

    def setup(self):

        self._var_names = {}

        rk_data = rk_methods[self.options['method']]
        num_stages = rk_data['num_stages']

        self.add_input('h', val=1.0, units=self.options['time_units'],
                       desc='step size for current Runge-Kutta segment.')

        for name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']
            rate_units = get_rate_units(units, self.options['time_units'])

            self._var_names[name] = {}
            self._var_names[name]['f'] = 'f:{0}'.format(name)
            self._var_names[name]['k'] = 'k:{0}'.format(name)

            self.add_input(self._var_names[name]['f'], shape=(num_stages,) + shape,
                           units=rate_units,
                           desc='The predicted values of the state at the ODE evaluation points.')

            self.add_output(self._var_names[name]['k'], shape=(num_stages,) + shape, units=units,
                            desc='RK multiplier k for each stage in the segment.')

            size = np.prod(shape)
            ar = np.arange(size * num_stages, dtype=int)
            self.declare_partials(of=self._var_names[name]['k'],
                                  wrt=self._var_names[name]['f'],
                                  rows=ar, cols=ar)

            r = np.arange(size * num_stages, dtype=int)
            c = np.zeros(size * num_stages, dtype=int)
            self.declare_partials(of=self._var_names[name]['k'],
                                  wrt='h',
                                  rows=r, cols=c)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h = inputs['h']
        for name, options in iteritems(self.options['state_options']):
            f = inputs[self._var_names[name]['f']]
            outputs[self._var_names[name]['k']] = f * h

    def compute_partials(self, inputs, partials):
        h = inputs['h']
        for name, options in iteritems(self.options['state_options']):
            partials[self._var_names[name]['k'], self._var_names[name]['f']] = h
            partials[self._var_names[name]['k'], 'h'] = inputs[self._var_names[name]['f']].ravel()
