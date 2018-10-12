from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import ImplicitComponent


class ImplicitSegmentConnectionComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('state_options', types=dict)

    def setup(self):

        self._vars = {}

        for state_name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']
            self._vars[state_name] = {}
            lhs_var = self._vars[state_name]['lhs'] = 'lhs_states:{0}'.format(state_name)
            rhs_var = self._vars[state_name]['rhs'] = 'rhs_states:{0}'.format(state_name)
            self.add_input(lhs_var, val=np.ones(shape), units=units)
            self.add_output(rhs_var, val=np.ones(shape), units=units)
            self.declare_partials(of=rhs_var, wrt=lhs_var, val=-np.ones(shape))

    def solve_nonlinear(self, inputs, outputs):
        for state_name, options in iteritems(self.options['state_options']):
            rhs_var = self._vars[state_name]['rhs']
            lhs_var = self._vars[state_name]['lhs']
            outputs[rhs_var] = inputs[lhs_var]

    def apply_nonlinear(self, inputs, outputs, residuals):
        for state_name, options in iteritems(self.options['state_options']):
            rhs_var = self._vars[state_name]['rhs']
            lhs_var = self._vars[state_name]['lhs']
            residuals[rhs_var] = outputs[rhs_var] - inputs[lhs_var]
