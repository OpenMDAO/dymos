from dymos.utils.misc import get_rate_units
import openmdao.api as om


class ODEIntControlInterpolationComp(om.ExplicitComponent):
    """
    Provides the interpolated value and rate of a control variable during explicit integration.

    For each control handled by ODEIntControlInterpolationComp, the user must provide an object
    with methods `eval(t)` and `eval_deriv(t)` which return the interpolated value and
    derivative of the control at time `t`, respectively.
    """
    def initialize(self):
        self.options.declare('time_units', default='s', allow_none=True, types=str,
                             desc='Units of time')
        self.options.declare('control_options', types=dict, allow_none=True, default=None,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('polynomial_control_options', types=dict, allow_none=True,
                             default=None, desc='Dictionary of options for the polynomial controls')
        self.control_interpolants = {}
        self.polynomial_control_interpolants = {}

    def setup(self):
        time_units = self.options['time_units']

        self.add_input('time', val=1.0, units=time_units)

        for control_name, options in self.options['control_options'].items():
            shape = options['shape']
            units = options['units']
            rate_units = get_rate_units(units, time_units, deriv=1)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self.add_output('controls:{0}'.format(control_name), shape=shape, units=units)

            self.add_output('control_rates:{0}_rate'.format(control_name), shape=shape,
                            units=rate_units)

            self.add_output('control_rates:{0}_rate2'.format(control_name), shape=shape,
                            units=rate2_units)

        for control_name, options in self.options['polynomial_control_options'].items():
            shape = options['shape']
            units = options['units']
            rate_units = get_rate_units(units, time_units, deriv=1)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self.add_output('polynomial_controls:{0}'.format(control_name), shape=shape,
                            units=units)

            self.add_output('polynomial_control_rates:{0}_rate'.format(control_name), shape=shape,
                            units=rate_units)

            self.add_output('polynomial_control_rates:{0}_rate2'.format(control_name), shape=shape,
                            units=rate2_units)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        time = inputs['time']

        for name in self.options['control_options']:
            if name not in self.control_interpolants:
                raise(ValueError('No interpolant has been specified for {0}'.format(name)))

            outputs['controls:{0}'.format(name)] = self.control_interpolants[name].eval(time)

            outputs['control_rates:{0}_rate'.format(name)] = \
                self.control_interpolants[name].eval_deriv(time)

            outputs['control_rates:{0}_rate2'.format(name)] = \
                self.control_interpolants[name].eval_deriv(time, der=2)

        for name in self.options['polynomial_control_options']:
            if name not in self.polynomial_control_interpolants:
                raise(ValueError('No interpolant has been specified for {0}'.format(name)))

            outputs['polynomial_controls:{0}'.format(name)] = \
                self.polynomial_control_interpolants[name].eval(time)

            outputs['polynomial_control_rates:{0}_rate'.format(name)] = \
                self.polynomial_control_interpolants[name].eval_deriv(time)

            outputs['polynomial_control_rates:{0}_rate2'.format(name)] = \
                self.polynomial_control_interpolants[name].eval_deriv(time, der=2)
