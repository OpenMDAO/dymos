import numpy as np
from openmdao.api import ExplicitComponent
from six import string_types, iteritems

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units


class CollocationComp(ExplicitComponent):

    """
    CollocationComp computes the generalized defect of a segment for implicit collocation.
    The defect is the interpolated state derivative at the collocation nodes minus
    the computed state derivative at the collocation nodes.
    """
    def initialize(self):

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')

    def setup(self):
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

        for state_name, options in iteritems(state_options):
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

            if 'defect_scaler' in options:
                def_scl = options['defect_scaler']
            else:
                def_scl = 1.0

            self.add_constraint(name=var_names['defect'],
                                equals=0.0,
                                scaler=def_scl)

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
        for state_name, options in iteritems(self.options['state_options']):
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]
            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            k = np.repeat(dt_dstau, size)

            partials[var_names['defect'], var_names['f_approx']] = k
            partials[var_names['defect'], var_names['f_computed']] = -k
            partials[var_names['defect'], 'dt_dstau'] = (f_approx - f_computed).ravel()
