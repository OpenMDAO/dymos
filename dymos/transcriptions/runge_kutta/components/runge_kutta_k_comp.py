import numpy as np

import openmdao.api as om
from ....utils.rk_methods import rk_methods
from ....utils.misc import get_rate_units
from ....options import options as dymos_options


class RungeKuttaKComp(om.ExplicitComponent):
    """
    Class definition for the RungeKuttaKComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('num_segments', types=int,
                             desc='The number of segments (timesteps) in the phase')

        self.options.declare('method', default='RK4', types=str,
                             desc='Specific Runge-Kutta Method to use.')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of the integration variable')

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        self._var_names = {}

        num_seg = self.options['num_segments']
        rk_data = rk_methods[self.options['method']]
        num_stages = rk_data['num_stages']

        self.add_input('h', val=np.ones(num_seg), units=self.options['time_units'],
                       desc='step size for current Runge-Kutta segment.')

        for name, options in self.options['state_options'].items():
            shape = options['shape']
            units = options['units']
            rate_units = get_rate_units(units, self.options['time_units'])

            self._var_names[name] = {}
            self._var_names[name]['f'] = 'f:{0}'.format(name)
            self._var_names[name]['k'] = 'k:{0}'.format(name)

            self.add_input(self._var_names[name]['f'], shape=(num_seg, num_stages) + shape,
                           units=rate_units,
                           desc='The predicted values of the state at the ODE evaluation points.')

            self.add_output(self._var_names[name]['k'], shape=(num_seg, num_stages) + shape,
                            units=units, desc='RK multiplier k for each stage in the segment.')

            size = np.prod(shape)
            ar = np.arange(size * num_stages * num_seg, dtype=int)
            self.declare_partials(of=self._var_names[name]['k'],
                                  wrt=self._var_names[name]['f'],
                                  rows=ar, cols=ar)

            r = np.arange(size * num_stages * num_seg, dtype=int)
            c = np.repeat(np.arange(num_seg, dtype=int), num_stages * size)
            self.declare_partials(of=self._var_names[name]['k'],
                                  wrt='h',
                                  rows=r, cols=c)

    def compute(self, inputs, outputs):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        h = inputs['h']
        for name, options in self.options['state_options'].items():
            f = inputs[self._var_names[name]['f']]
            outputs[self._var_names[name]['k']] = f * h[:, np.newaxis, np.newaxis]

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        num_stages = rk_methods[self.options['method']]['num_stages']
        h = inputs['h']
        for name, options in self.options['state_options'].items():
            size = np.prod(options['shape'])
            k_name = self._var_names[name]['k']
            f_name = self._var_names[name]['f']
            partials[k_name, f_name] = np.repeat(h, num_stages * size)
            partials[k_name, 'h'] = inputs[self._var_names[name]['f']].ravel()
