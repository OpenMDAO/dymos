"""Define the CollocationComp class."""
import numpy as np

import openmdao.api as om

from ...grid_data import GridData
from ....utils.misc import get_rate_units
from ....options import options as dymos_options


class CollocationComp(om.ExplicitComponent):
    """
    CollocationComp computes the generalized defect of a segment for implicit collocation.
    The defect is the interpolated state derivative at the collocation nodes minus
    the computed state derivative at the collocation nodes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        gd = self.options['grid_data']
        num_col_nodes = gd.subset_num_nodes['col']
        time_units = self.options['time_units']
        state_options = self.options['state_options']

        self.add_input('dt_dstau', units=time_units, shape=(num_col_nodes,))

        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'f_approx': 'f_approx:{0}'.format(state_name),
                'f_computed': 'f_computed:{0}'.format(state_name),
                'defect': 'defects:{0}'.format(state_name),
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            var_names = self.var_names[state_name]

            self.add_input(
                name=var_names['f_approx'],
                shape=(num_col_nodes,) + shape,
                desc='Estimated derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)

            self.add_input(
                name=var_names['f_computed'],
                shape=(num_col_nodes,) + shape,
                desc='Computed derivative of state {0} '
                     'at the collocation nodes'.format(state_name),
                units=rate_units)

            self.add_output(
                name=var_names['defect'],
                shape=(num_col_nodes,) + shape,
                desc='Interior defects of state {0}'.format(state_name),
                units=units)

            if 'defect_ref' in options and options['defect_ref'] is not None:
                defect_ref = options['defect_ref']
            elif 'defect_scaler' in options and options['defect_scaler'] is not None:
                defect_ref = 1.0 / options['defect_scaler']
            else:
                if 'ref' in options and options['ref'] is not None:
                    defect_ref = options['ref']
                elif 'scaler' in options and options['scaler'] is not None:
                    defect_ref = 1.0 / options['scaler']
                else:
                    defect_ref = 1.0

            if not options['solve_segments']:
                self.add_constraint(name=var_names['defect'],
                                    equals=0.0,
                                    ref=defect_ref)

        # Setup partials
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']
        state_options = self.options['state_options']

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)

            r = np.arange(num_col_nodes * size)

            var_names = self.var_names[state_name]

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_approx'],
                                  rows=r, cols=r)

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_computed'],
                                  rows=r, cols=r)

            c = np.repeat(np.arange(num_col_nodes), size)
            self.declare_partials(of=var_names['defect'],
                                  wrt='dt_dstau',
                                  rows=r, cols=c)

    def compute(self, inputs, outputs):
        state_options = self.options['state_options']
        dt_dstau = inputs['dt_dstau']

        for state_name in state_options:
            var_names = self.var_names[state_name]

            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            outputs[var_names['defect']] = ((f_approx - f_computed).T * dt_dstau).T

    def compute_partials(self, inputs, partials):
        dt_dstau = inputs['dt_dstau']
        for state_name, options in self.options['state_options'].items():
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]
            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            k = np.repeat(dt_dstau, size)

            partials[var_names['defect'], var_names['f_approx']] = k
            partials[var_names['defect'], var_names['f_computed']] = -k
            partials[var_names['defect'], 'dt_dstau'] = (f_approx - f_computed).ravel()
