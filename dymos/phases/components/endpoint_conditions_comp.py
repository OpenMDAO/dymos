from __future__ import print_function, division, absolute_import

from six import iteritems
import numpy as np
from openmdao.api import ExplicitComponent, OptionsDictionary


class EndpointConditionsComp(ExplicitComponent):
    """
    Provides initial conditions for the states and times
    of a phase that can be used to constrain initial bounds or enforce
    phase continuity.
    """
    def initialize(self):
        self.metadata.declare('time_options', types=OptionsDictionary)
        self.metadata.declare('state_options', types=dict)
        self.metadata.declare('control_options', types=dict)

    def _setup_states(self):
        for name, options in iteritems(self.metadata['state_options']):
            shape = options['shape']
            units = options['units']

            self.add_input(
                name='initial_value:{0}'.format(name),
                val=np.zeros(shape),
                desc='The initial value of state {0}'.format(name),
                units=units)

            self.add_input(
                name='final_value:{0}'.format(name),
                val=np.zeros(shape),
                desc='The final value of state {0}'.format(name),
                units=units)

            self.add_input(
                name='initial_jump:{0}'.format(name),
                val=np.zeros(shape),
                desc='The discontinuity in state {0} at the start of the phase'.format(name),
                units=units)

            self.add_input(
                name='final_jump:{0}'.format(name),
                val=np.zeros(shape),
                desc='The discontinuity in state {0} at the end of the phase'.format(name),
                units=units)

            self.add_output(
                name='states:{0}--'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} before the initial jump'.format(name),
                units=units)

            self.add_output(
                name='states:{0}-+'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} after the initial jump'.format(name),
                units=units)

            self.add_output(
                name='states:{0}++'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} after the final jump'.format(name),
                units=units)

            self.add_output(
                name='states:{0}+-'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} before the final jump'.format(name),
                units=units)

            size = np.prod(shape)
            ar = np.arange(size)

            self.declare_partials(of='states:{0}--'.format(name),
                                  wrt='initial_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}--'.format(name),
                                  wrt='initial_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=-1.0)

            self.declare_partials(of='states:{0}-+'.format(name),
                                  wrt='initial_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}+-'.format(name),
                                  wrt='final_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}++'.format(name),
                                  wrt='final_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}++'.format(name),
                                  wrt='final_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=1.0)

    def _setup_controls(self):
        for name, options in iteritems(self.metadata['control_options']):
            shape = options['shape']
            units = options['units']

            self.add_input(
                name='initial_value:{0}'.format(name),
                val=np.zeros(shape),
                desc='The initial value of state {0}'.format(name),
                units=units)

            self.add_input(
                name='final_value:{0}'.format(name),
                val=np.zeros(shape),
                desc='The final value of state {0}'.format(name),
                units=units)

            self.add_input(
                name='initial_jump:{0}'.format(name),
                val=np.zeros(shape),
                desc='The discontinuity in state {0} at the start of the phase'.format(name),
                units=units)

            self.add_input(
                name='final_jump:{0}'.format(name),
                val=np.zeros(shape),
                desc='The discontinuity in state {0} at the end of the phase'.format(name),
                units=units)

            self.add_output(
                name='controls:{0}--'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} before the initial jump'.format(name),
                units=units)

            self.add_output(
                name='controls:{0}-+'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} after the initial jump'.format(name),
                units=units)

            self.add_output(
                name='controls:{0}++'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} after the final jump'.format(name),
                units=units)

            self.add_output(
                name='controls:{0}+-'.format(name),
                val=np.zeros(shape),
                desc='The value of state {0} before the final jump'.format(name),
                units=units)

            size = np.prod(shape)
            ar = np.arange(size)

            self.declare_partials(of='controls:{0}--'.format(name),
                                  wrt='initial_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}--'.format(name),
                                  wrt='initial_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=-1.0)

            self.declare_partials(of='controls:{0}-+'.format(name),
                                  wrt='initial_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}+-'.format(name),
                                  wrt='final_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}++'.format(name),
                                  wrt='final_value:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}++'.format(name),
                                  wrt='final_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=1.0)

    def _setup_time(self):
        time_units = self.metadata['time_options']['units']

        self.add_input(name='initial_value:time',
                       desc='value of time at the beginning of integration',
                       val=np.zeros(1),
                       units=time_units)

        self.add_input(name='initial_jump:time',
                       desc='discontinuity in time at the beginning of the phase',
                       val=np.zeros(1),
                       units=time_units)

        self.add_input(name='final_value:time',
                       desc='value of time at the end of integration',
                       val=np.zeros(1),
                       units=time_units)

        self.add_input(name='final_jump:time',
                       desc='discontinuity in time at the end of the phase',
                       val=np.zeros(1),
                       units=time_units)

        self.add_output(name='time--',
                        desc='value of time before the initial jump',
                        val=np.zeros(1),
                        units=time_units)

        self.add_output(name='time-+',
                        desc='the value of time at the beginning of integration',
                        val=np.zeros(1),
                        units=time_units)

        self.add_output(name='time++',
                        desc='value of time after the final jump',
                        val=np.zeros(1),
                        units=time_units)

        self.add_output(name='time+-',
                        desc='the value of time at the end of integration',
                        val=np.zeros(1),
                        units=time_units)

        self.declare_partials(of='time--', wrt='initial_value:time', val=1.0)
        self.declare_partials(of='time--', wrt='initial_jump:time', val=-1.0)
        self.declare_partials(of='time-+', wrt='initial_value:time', val=1.0)
        self.declare_partials(of='time+-', wrt='final_value:time', val=1.0)
        self.declare_partials(of='time++', wrt='final_value:time', val=1.0)
        self.declare_partials(of='time++', wrt='final_jump:time', val=1.0)

    def setup(self):
        self._setup_time()
        self._setup_states()
        self._setup_controls()

    def compute(self, inputs, outputs):

        outputs['time--'] = inputs['initial_value:time'] - inputs['initial_jump:time']
        outputs['time-+'] = inputs['initial_value:time']
        outputs['time+-'] = inputs['final_value:time']
        outputs['time++'] = inputs['final_value:time'] + inputs['final_jump:time']

        for state_name, options in iteritems(self.metadata['state_options']):
            outputs['states:{0}--'.format(state_name)] = \
                inputs['initial_value:{0}'.format(state_name)] - \
                inputs['initial_jump:{0}'.format(state_name)]

            outputs['states:{0}-+'.format(state_name)] = \
                inputs['initial_value:{0}'.format(state_name)]

            outputs['states:{0}+-'.format(state_name)] = \
                inputs['final_value:{0}'.format(state_name)]

            outputs['states:{0}++'.format(state_name)] = \
                inputs['final_value:{0}'.format(state_name)] + \
                inputs['final_jump:{0}'.format(state_name)]

        for control_name, options in iteritems(self.metadata['control_options']):
            outputs['controls:{0}--'.format(control_name)] = \
                inputs['initial_value:{0}'.format(control_name)] - \
                inputs['initial_jump:{0}'.format(control_name)]

            outputs['controls:{0}-+'.format(control_name)] = \
                inputs['initial_value:{0}'.format(control_name)]

            outputs['controls:{0}+-'.format(control_name)] = \
                inputs['final_value:{0}'.format(control_name)]

            outputs['controls:{0}++'.format(control_name)] = \
                inputs['final_value:{0}'.format(control_name)] + \
                inputs['final_jump:{0}'.format(control_name)]
