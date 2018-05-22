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
        self.options.declare('time_options', types=OptionsDictionary)
        self.options.declare('state_options', types=dict)
        self.options.declare('control_options', types=dict)

    def _setup_states(self):
        for name, options in iteritems(self.options['state_options']):
            shape = options['shape']
            units = options['units']
            size = np.prod(shape)

            src_idxs_start = np.arange(0, size, 1, dtype=int)
            src_idxs_end = np.arange(-size, 0, 1, dtype=int)
            src_idxs = np.reshape(np.concatenate((src_idxs_start, src_idxs_end)),
                                  newshape=(2,) + shape)
            self.add_input(
                name='values:{0}'.format(name),
                val=np.zeros_like(src_idxs),
                desc='values of state {0} at the endpoints of the phase'.format(name),
                units=units,
                src_indices=src_idxs,
                flat_src_indices=True)

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

            ar = np.arange(size)

            self.declare_partials(of='states:{0}--'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}--'.format(name),
                                  wrt='initial_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=-1.0)

            self.declare_partials(of='states:{0}-+'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}+-'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=size + ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}++'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=size + ar,
                                  val=1.0)

            self.declare_partials(of='states:{0}++'.format(name),
                                  wrt='final_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=1.0)

    def _setup_controls(self):
        for name, options in iteritems(self.options['control_options']):
            shape = options['shape']
            units = options['units']
            size = np.prod(shape)

            src_idxs_start = np.arange(0, size, 1, dtype=int)
            src_idxs_end = np.arange(-size, 0, 1, dtype=int)
            src_idxs = np.reshape(np.concatenate((src_idxs_start, src_idxs_end)),
                                  newshape=(2,) + shape)

            self.add_input(
                name='values:{0}'.format(name),
                val=np.zeros_like(src_idxs),
                desc='Values of control {0} at the phase endpoints'.format(name),
                units=units,
                src_indices=src_idxs,
                flat_src_indices=True)

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

            ar = np.arange(size)

            self.declare_partials(of='controls:{0}--'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}--'.format(name),
                                  wrt='initial_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=-1.0)

            self.declare_partials(of='controls:{0}-+'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}+-'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=size + ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}++'.format(name),
                                  wrt='values:{0}'.format(name), rows=ar, cols=size + ar,
                                  val=1.0)

            self.declare_partials(of='controls:{0}++'.format(name),
                                  wrt='final_jump:{0}'.format(name), rows=ar,
                                  cols=ar, val=1.0)

    def _setup_time(self):
        time_units = self.options['time_options']['units']

        self.add_input(name='values:time',
                       desc='value of time at the endpoints of the phase',
                       val=np.zeros(2),
                       units=time_units,
                       src_indices=[0, -1],
                       flat_src_indices=True)

        self.add_input(name='initial_jump:time',
                       desc='discontinuity in time at the beginning of the phase',
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

        self.declare_partials(of='time--', wrt='values:time', val=np.array([[1.0, 0]]))
        self.declare_partials(of='time--', wrt='initial_jump:time', val=-1.0)
        self.declare_partials(of='time-+', wrt='values:time', val=np.array([[1.0, 0]]))
        self.declare_partials(of='time+-', wrt='values:time', val=np.array([[0, 1.0]]))
        self.declare_partials(of='time++', wrt='values:time', val=np.array([[0, 1.0]]))
        self.declare_partials(of='time++', wrt='final_jump:time', val=1.0)

    def setup(self):
        self._setup_time()
        self._setup_states()
        self._setup_controls()

    def compute(self, inputs, outputs):

        outputs['time--'] = inputs['values:time'][0] - inputs['initial_jump:time']
        outputs['time-+'] = inputs['values:time'][0]
        outputs['time+-'] = inputs['values:time'][-1]
        outputs['time++'] = inputs['values:time'][-1] + inputs['final_jump:time']

        for state_name, options in iteritems(self.options['state_options']):
            outputs['states:{0}--'.format(state_name)] = \
                inputs['values:{0}'.format(state_name)][0, ...] - \
                inputs['initial_jump:{0}'.format(state_name)]

            outputs['states:{0}-+'.format(state_name)] = \
                inputs['values:{0}'.format(state_name)][0, ...]

            outputs['states:{0}+-'.format(state_name)] = \
                inputs['values:{0}'.format(state_name)][-1, ...]

            outputs['states:{0}++'.format(state_name)] = \
                inputs['values:{0}'.format(state_name)][-1, ...] + \
                inputs['final_jump:{0}'.format(state_name)]

        for control_name, options in iteritems(self.options['control_options']):
            outputs['controls:{0}--'.format(control_name)] = \
                inputs['values:{0}'.format(control_name)][0, ...] - \
                inputs['initial_jump:{0}'.format(control_name)]

            outputs['controls:{0}-+'.format(control_name)] = \
                inputs['values:{0}'.format(control_name)][0, ...]

            outputs['controls:{0}+-'.format(control_name)] = \
                inputs['values:{0}'.format(control_name)][-1, ...]

            outputs['controls:{0}++'.format(control_name)] = \
                inputs['values:{0}'.format(control_name)][-1, ...] + \
                inputs['final_jump:{0}'.format(control_name)]
