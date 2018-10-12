from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from openmdao.api import ExplicitComponent


class ImplicitSegmentConnectionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('state_options', types=dict)

    def setup(self):

        self._vars = {}

        for state_name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']
            lhs_var = self._vars[state_name]['lhs'] = 'lhs_states:{0}'.format(state_name)
            rhs_var = self._vars[state_name]['rhs'] = 'rhs_states:{0}'.format(state_name)
            def_var = self._vars[state_name]['defect'] = 'defects:{0}'.format(state_name)
            self.add_input(lhs_var, val=np.ones(shape), units=units)
            self.add_input(rhs_var, val=np.ones(shape), units=units)
            self.add_output(def_var, val=np.ones(shape), units=units)
            self.declare_partials(of=def_var, wrt=lhs_var, val=-np.ones(shape))
            self.declare_partials(of=def_var, wrt=rhs_var, val=np.ones(shape))

    def compute(self, inputs, outputs):
        for state_name, options in iteritems(self.options['state_options']):
            rhs_var = self._vars[state_name]['rhs']
            lhs_var = self._vars[state_name]['lhs']
            def_var = self._vars[state_name]['defect']
            outputs[def_var] = inputs[rhs_var] - inputs[lhs_var]

    def apply_nonlinear(self, inputs, outputs, residuals):
        for state_name, options in iteritems(self.options['state_options']):
            rhs_var = self._vars[state_name]['rhs']
            lhs_var = self._vars[state_name]['lhs']
            residuals[rhs_var] = outputs[rhs_var] - inputs[lhs_var]
