import numpy as np
import openmdao.api as om


class EndpointConditionsComp(om.ExplicitComponent):
    """
    Provides values of time, states, and controls at the start/end of each
    phase to make it simpler to link phases together.
    """
    def initialize(self):
        self.options.declare('loc', values=('initial', 'final'),
                             desc='Whether the instance of the component provides conditions at '
                                  'the start (initial) or end (final) of the phase')
        self.options.declare('time_options', types=om.OptionsDictionary)
        self.options.declare('state_options', types=dict)
        self.options.declare('control_options', types=dict)

    def _setup_states(self):
        loc = self.options['loc']
        for name, options in self.options['state_options'].items():
            shape = options['shape']
            units = options['units']
            size = int(np.prod(shape))
            ar = np.arange(size, dtype=int)

            if loc == 'initial':
                src_idxs = np.arange(0, size, 1, dtype=int)
            else:
                src_idxs = np.arange(-size, 0, 1, dtype=int)

            self.add_input(
                name='{0}_value:{1}'.format(loc, name),
                val=np.zeros_like(src_idxs),
                desc='{0} values of state {1} in the phase'.format(loc, name),
                units=units,
                src_indices=src_idxs,
                flat_src_indices=True)

            self.add_input(
                name='{0}_jump:{1}'.format(loc, name),
                val=np.zeros(shape),
                desc='{0} discontinuity in state {1} at the start of the phase'.format(loc, name),
                units=units)

            # self.add_input(
            #     name='final_jump:{0}'.format(name),
            #     val=np.zeros(shape),
            #     desc='The discontinuity in state {0} at the end of the phase'.format(name),
            #     units=units)

            if loc == 'initial':
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

                self.declare_partials(of='states:{0}--'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

                self.declare_partials(of='states:{0}--'.format(name),
                                      wrt='initial_jump:{0}'.format(name), rows=ar,
                                      cols=ar, val=-1.0)

                self.declare_partials(of='states:{0}-+'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

            else:
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

                self.declare_partials(of='states:{0}+-'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

                self.declare_partials(of='states:{0}++'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

                self.declare_partials(of='states:{0}++'.format(name),
                                      wrt='final_jump:{0}'.format(name), rows=ar,
                                      cols=ar, val=1.0)

    def _setup_controls(self):
        loc = self.options['loc']
        for name, options in self.options['control_options'].items():
            shape = options['shape']
            units = options['units']
            size = int(np.prod(shape))
            ar = np.arange(size)

            if loc == 'initial':
                src_idxs = np.arange(0, size, 1, dtype=int)
            else:
                src_idxs = np.arange(-size, 0, 1, dtype=int)

            self.add_input(
                name='{0}_value:{1}'.format(loc, name),
                val=np.zeros_like(src_idxs),
                desc='{0} values of state {1} in the phase'.format(loc, name),
                units=units,
                src_indices=src_idxs,
                flat_src_indices=True)

            self.add_input(
                name='{0}_jump:{1}'.format(loc, name),
                val=np.zeros(shape),
                desc='{0} discontinuity in state {1} at the start of the phase'.format(loc, name),
                units=units)

            # self.add_input(
            #     name='values:{0}'.format(name),
            #     val=np.zeros_like(src_idxs),
            #     desc='Values of control {0} at the phase endpoints'.format(name),
            #     units=units,
            #     src_indices=src_idxs,
            #     flat_src_indices=True)
            #
            # self.add_input(
            #     name='initial_jump:{0}'.format(name),
            #     val=np.zeros(shape),
            #     desc='The discontinuity in state {0} at the start of the phase'.format(name),
            #     units=units)
            #
            # self.add_input(
            #     name='final_jump:{0}'.format(name),
            #     val=np.zeros(shape),
            #     desc='The discontinuity in state {0} at the end of the phase'.format(name),
            #     units=units)

            if loc == 'initial':
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

                self.declare_partials(of='controls:{0}--'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

                self.declare_partials(of='controls:{0}--'.format(name),
                                      wrt='initial_jump:{0}'.format(name), rows=ar,
                                      cols=ar, val=-1.0)

                self.declare_partials(of='controls:{0}-+'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

            else:
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

                self.declare_partials(of='controls:{0}+-'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

                self.declare_partials(of='controls:{0}++'.format(name),
                                      wrt='{0}_value:{1}'.format(loc, name), rows=ar, cols=ar,
                                      val=1.0)

                self.declare_partials(of='controls:{0}++'.format(name),
                                      wrt='final_jump:{0}'.format(name), rows=ar,
                                      cols=ar, val=1.0)

    def _setup_time(self):
        loc = self.options['loc']
        time_units = self.options['time_options']['units']

        src_idxs = [0] if loc == 'initial' else [-1]

        self.add_input(name='{0}_value:time'.format(loc),
                       desc='{0} value of time in the phase'.format(loc),
                       val=np.zeros(1),
                       units=time_units,
                       src_indices=src_idxs,
                       flat_src_indices=True)

        self.add_input(name='{0}_jump:time'.format(loc),
                       desc='{0} discontinuity in time for the phase'.format(loc),
                       val=np.zeros(1),
                       units=time_units)

        # self.add_input(name='final_jump:time',
        #                desc='discontinuity in time at the end of the phase',
        #                val=np.zeros(1),
        #                units=time_units)

        if loc == 'initial':
            self.add_output(name='time--',
                            desc='value of time before the initial jump',
                            val=np.zeros(1),
                            units=time_units)

            self.add_output(name='time-+',
                            desc='the value of time at the beginning of integration',
                            val=np.zeros(1),
                            units=time_units)

            self.declare_partials(of='time--', wrt='{0}_value:time'.format(loc), val=1.0)
            self.declare_partials(of='time--', wrt='initial_jump:time', val=-1.0)
            self.declare_partials(of='time-+', wrt='{0}_value:time'.format(loc), val=1.0)

        else:
            self.add_output(name='time++',
                            desc='value of time after the final jump',
                            val=np.zeros(1),
                            units=time_units)

            self.add_output(name='time+-',
                            desc='the value of time at the end of integration',
                            val=np.zeros(1),
                            units=time_units)

            self.declare_partials(of='time+-', wrt='{0}_value:time'.format(loc), val=1.0)
            self.declare_partials(of='time++', wrt='{0}_value:time'.format(loc), val=1.0)
            self.declare_partials(of='time++', wrt='final_jump:time', val=1.0)

    def setup(self):
        self._setup_time()
        self._setup_states()
        self._setup_controls()

    def compute(self, inputs, outputs):
        loc = self.options['loc']

        if loc == 'initial':
            outputs['time--'] = inputs['initial_value:time'][0] - inputs['initial_jump:time']
            outputs['time-+'] = inputs['initial_value:time'][0]
        else:
            outputs['time+-'] = inputs['final_value:time'][0]
            outputs['time++'] = inputs['final_value:time'][0] + inputs['final_jump:time']

        for state_name, options in self.options['state_options'].items():
            if loc == 'initial':
                outputs['states:{0}--'.format(state_name)] = \
                    inputs['initial_value:{0}'.format(state_name)] - \
                    inputs['initial_jump:{0}'.format(state_name)]

                outputs['states:{0}-+'.format(state_name)] = \
                    inputs['initial_value:{0}'.format(state_name)]

            else:
                outputs['states:{0}+-'.format(state_name)] = \
                    inputs['final_value:{0}'.format(state_name)]

                outputs['states:{0}++'.format(state_name)] = \
                    inputs['final_value:{0}'.format(state_name)] + \
                    inputs['final_jump:{0}'.format(state_name)]

        for control_name, options in self.options['control_options'].items():

            if loc == 'initial':
                outputs['controls:{0}--'.format(control_name)] = \
                    inputs['initial_value:{0}'.format(control_name)] - \
                    inputs['initial_jump:{0}'.format(control_name)]

                outputs['controls:{0}-+'.format(control_name)] = \
                    inputs['initial_value:{0}'.format(control_name)]

            else:
                outputs['controls:{0}+-'.format(control_name)] = \
                    inputs['final_value:{0}'.format(control_name)]

                outputs['controls:{0}++'.format(control_name)] = \
                    inputs['final_value:{0}'.format(control_name)] + \
                    inputs['final_jump:{0}'.format(control_name)]
