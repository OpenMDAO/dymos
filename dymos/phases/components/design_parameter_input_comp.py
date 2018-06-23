from __future__ import division, print_function

import numpy as np
from six import iteritems

from openmdao.api import ExplicitComponent


class DesignParameterInputComp(ExplicitComponent):
    """
    The DesignParameterInputComp handles design parameters in the phase for which `opt==False`.

    This component makes the values of the design parameter available to be connected to an
    external source.
    """
    def initialize(self):
        self.options.declare(name='num_nodes',
                             types=int,
                             desc='The number of nodes in the phase')

        self.options.declare(name='design_parameter_options',
                             types=dict,
                             desc='Dictionary of options for the design parameters')

        self._input_design_parameters = []
        self._output_names = {}
        self._input_names = {}

    def setup(self):
        for param_name, options in iteritems(self.options['design_parameter_options']):
            if options['opt']:
                # Ignore this control if it is an optimal control
                continue
            self._input_design_parameters.append(param_name)
            self._input_names[param_name] = 'design_parameters:{0}'.format(param_name)
            self._output_names[param_name] = 'design_parameters:{0}_out'.format(param_name)

            n = 1  # the design parameter effectively has a value at a single node

            shape = (n,) + options['shape']
            units = options['units']
            size = np.prod(shape)

            if 'val' in options:
                val = np.atleast_1d(np.asarray(options['val']))

                if len(val) == 1:
                    default_val = val * np.ones(shape)
                elif val.shape == shape[1:]:
                    default_val = np.zeros(shape)
                    default_val[...] = val[np.newaxis, ...]
                else:
                    raise ValueError('Default value of {0} has an incompatible shape.  Have {1} but'
                                     ' need {2}'.format(param_name, val.shape, shape))
            else:
                default_val = np.ones((n,) + shape)

            self.add_input(self._input_names[param_name], val=default_val,
                           units=units, shape=shape)

            self.add_output(self._output_names[param_name], val=default_val,
                            units=units, shape=shape)

            arange = np.arange(size)
            self.declare_partials(of=self._output_names[param_name],
                                  wrt=self._input_names[param_name],
                                  val=np.ones(size), rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        for param_name in self._input_design_parameters:
            outputs[self._output_names[param_name]] = inputs[self._input_names[param_name]]
