from __future__ import division, print_function

import numpy as np
from six import iteritems

import openmdao.api as om


class InputParameterComp(om.ExplicitComponent):
    """
    The InputParameterComp handles input parameters for phases and trajectories.

    This component makes the values of the design parameter available to be connected to an
    external source.
    """
    def initialize(self):
        self.options.declare(name='input_parameter_options',
                             types=dict,
                             desc='Dictionary of options for the input parameters')

        self.options.declare(name='traj_params', types=bool, default=False,
                             desc='True if this input parameter comp is for trajectory parameters.')

        self._input_design_parameters = []
        self._output_names = {}
        self._input_names = {}

    def setup(self):
        name_prefix = 'input_parameters'

        for param_name, options in iteritems(self.options['input_parameter_options']):

            self._input_design_parameters.append(param_name)
            self._input_names[param_name] = '{0}:{1}'.format(name_prefix, param_name)
            self._output_names[param_name] = '{0}:{1}_out'.format(name_prefix, param_name)

            n = 1

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
