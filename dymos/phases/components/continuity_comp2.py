from __future__ import print_function, division

import numpy as np
from openmdao.api import ExplicitComponent
from six import iteritems, string_types

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units


class ContinuityComp2(ExplicitComponent):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.
    """

    def initialize(self):

        self.options.declare('grid_data', types=GridData,
                             desc='Container object for grid info')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

        self.options.declare('control_options', types=dict,
                             desc='Dictionary of control names/options for the phase')

        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of the integration variable')

        self.options.declare('enforce_state_continuity', default=True, types=bool,
                             desc='Whether to add state continuity constraints')

    def _setup_value_continuity(self):
        state_options = self.options['state_options']
        control_options = self.options['control_options']
        num_segend_nodes = self.options['grid_data'].subset_num_nodes['segment_ends']
        num_segments = self.options['grid_data'].num_segments

        self.rate_jac_templates = {}

        for state_name, options in iteritems(state_options):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            self.add_input(name='states:{0}'.format(state_name),
                           shape=(num_segend_nodes,) + shape,
                           desc='Values of state {0} at discretization nodes'.format(state_name),
                           units=units)

            self.add_output(
                name='defect_states:{0}'.format(state_name),
                shape=(num_segments - 1,) + shape,
                desc='Consistency constraint values for state {0}'.format(state_name),
                units=units)

            if self.options['enforce_state_continuity']:
                self.add_constraint(name='defect_states:{0}'.format(state_name),
                                    equals=0.0, scaler=1.0, linear=True)

            rs_size1 = np.repeat(np.arange(num_segments-1, dtype=int), 2)
            cs_size1 = np.arange(1, num_segend_nodes-1, dtype=int)

            template = np.zeros((num_segments-1, num_segend_nodes))
            template[rs_size1, cs_size1] = 1.0
            template = np.kron(template, np.eye(size))
            rs, cs = template.nonzero()

            vals = np.zeros(len(rs), dtype=float)
            vals[0::2] = -1.0
            vals[1::2] = 1.0

            self.declare_partials(
                'defect_states:{0}'.format(state_name),
                'states:{0}'.format(state_name),
                val=vals, rows=rs, cols=cs,
            )

        for control_name, options in iteritems(control_options):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            if options['dynamic'] and options['continuity']:
                self.add_input(
                    name='controls:{0}'.format(control_name),
                    shape=(num_segend_nodes,) + shape,
                    desc='Values of control {0} at discretization nodes'.format(control_name),
                    units=units)

                self.add_output(
                    name='defect_controls:{0}'.format(control_name),
                    shape=(num_segments - 1,) + shape,
                    desc='Consistency constraint values for control {0}'.format(control_name),
                    units=units)

                self.add_constraint(name='defect_controls:{0}'.format(control_name),
                                    equals=0.0, scaler=1.0, linear=True)

                rs_size1 = np.repeat(np.arange(num_segments - 1, dtype=int), 2)
                cs_size1 = np.arange(1, num_segend_nodes - 1, dtype=int)

                template = np.zeros((num_segments - 1, num_segend_nodes))
                template[rs_size1, cs_size1] = 1.0
                template = np.kron(template, np.eye(size))
                rs, cs = template.nonzero()

                vals = np.zeros(len(rs), dtype=float)
                vals[0::2] = -1.0
                vals[1::2] = 1.0

                self.declare_partials(
                    'defect_controls:{0}'.format(control_name),
                    'controls:{0}'.format(control_name),
                    val=vals, rows=rs, cols=cs,
                )

    def _setup_rate_continuity(self):
        control_options = self.options['control_options']
        num_segend_nodes = self.options['grid_data'].subset_num_nodes['segment_ends']
        num_segments = self.options['grid_data'].num_segments
        time_units = self.options['time_units']

        self.add_input('t_duration', units=time_units, val=1.0, desc='time duration of the phase')

        for control_name, options in iteritems(control_options):
            shape = options['shape']
            units = options['units']
            rate_units = get_rate_units(units, self.options['time_units'], deriv=1)
            rate2_units = get_rate_units(units, self.options['time_units'], deriv=2)

            if options['opt'] and options['dynamic']:

                size = np.prod(shape)

                # Define the sparsity pattern for rate and rate2 continuity
                rs_size1 = np.repeat(np.arange(num_segments - 1, dtype=int), 2)
                cs_size1 = np.arange(1, num_segend_nodes - 1, dtype=int)

                template = np.zeros((num_segments - 1, num_segend_nodes))
                template[rs_size1, cs_size1] = 1.0
                template = np.kron(template, np.eye(size))
                rs, cs = template.nonzero()

                vals = np.zeros(len(rs), dtype=float)
                vals[0::2] = -1.0
                vals[1::2] = 1.0
                self.rate_jac_templates[control_name] = vals

                if options['rate_continuity']:
                    self.add_input(
                        name='control_rates:{0}_rate'.format(control_name),
                        shape=(num_segend_nodes,) + shape,
                        desc='Values of control {0} derivative at '
                             'discretization nodes'.format(control_name),
                        units=rate_units)

                    self.add_output(
                        name='defect_control_rates:{0}_rate'.format(control_name),
                        shape=(num_segments - 1,) + shape,
                        desc='Consistency constraint values for '
                             'control {0} derivative'.format(control_name),
                        units=rate_units)

                    self.add_constraint(name='defect_control_rates:{0}_rate'.format(control_name),
                                        equals=0.0, scaler=options['rate_continuity_scaler'],
                                        linear=False)

                    self.declare_partials(
                        'defect_control_rates:{0}_rate'.format(control_name),
                        'control_rates:{0}_rate'.format(control_name),
                        rows=rs, cols=cs,
                    )

                    self.declare_partials(
                        'defect_control_rates:{0}_rate'.format(control_name),
                        't_duration', dependent=True,
                    )

                if options['rate2_continuity']:
                    self.add_input(
                        name='control_rates:{0}_rate2'.format(control_name),
                        shape=(num_segend_nodes,) + shape,
                        desc='Values of control {0} second derivative '
                             'at discretization nodes'.format(control_name),
                        units=rate2_units)

                    self.add_output(
                        name='defect_control_rates:{0}_rate2'.format(control_name),
                        shape=(num_segments - 1,) + shape,
                        desc='Consistency constraint values for control '
                             '{0} second derivative'.format(control_name),
                        units=rate2_units)

                    self.add_constraint(name='defect_control_rates:{0}_rate2'.format(control_name),
                                        equals=0.0, scaler=options['rate2_continuity_scaler'],
                                        linear=False)

                    self.declare_partials(
                        'defect_control_rates:{0}_rate2'.format(control_name),
                        'control_rates:{0}_rate2'.format(control_name),
                        rows=rs, cols=cs,
                    )

                    self.declare_partials(
                        'defect_control_rates:{0}_rate2'.format(control_name),
                        't_duration', dependent=True
                    )

    def setup(self):
        self.rate_jac_templates = {}
        compressed = self.options['grid_data'].compressed
        if not compressed:
            self._setup_value_continuity()
        self._setup_rate_continuity()

    def compute(self, inputs, outputs):
        state_options = self.options['state_options']
        control_options = self.options['control_options']
        compressed = self.options['grid_data'].compressed

        if not compressed:
            for state_name, options in iteritems(state_options):

                end_vals = inputs['states:{0}'.format(state_name)][1:-1:2, ...]
                start_vals = inputs['states:{0}'.format(state_name)][2:-1:2, ...]

                outputs['defect_states:{0}'.format(state_name)] = start_vals - end_vals

            for name, options in iteritems(control_options):
                if options['dynamic'] and options['continuity']:
                    end_vals = inputs['controls:{0}'.format(name)][1:-1:2, ...]
                    start_vals = inputs['controls:{0}'.format(name)][2:-1:2, ...]

                    outputs['defect_controls:{0}'.format(name)] = start_vals - end_vals

        dt_dptau = inputs['t_duration'] / 2.0

        for name, options in iteritems(control_options):
            if options['opt'] and options['dynamic']:
                if options['rate_continuity']:
                    end_vals = inputs['control_rates:{0}_rate'.format(name)][1:-1:2, ...]
                    start_vals = inputs['control_rates:{0}_rate'.format(name)][2:-1:2, ...]

                    outputs['defect_control_rates:{0}_rate'.format(name)] = \
                        (start_vals - end_vals) * dt_dptau

                if options['rate2_continuity']:
                    end_vals = inputs['control_rates:{0}_rate2'.format(name)][1:-1:2, ...]
                    start_vals = inputs['control_rates:{0}_rate2'.format(name)][2:-1:2, ...]

                    outputs['defect_control_rates:{0}_rate2'.format(name)] = \
                        (start_vals - end_vals) * dt_dptau ** 2

    def compute_partials(self, inputs, partials):

        control_options = self.options['control_options']
        dt_dptau = 0.5 * inputs['t_duration']

        for control_name, options in iteritems(control_options):
            if options['opt'] and options['dynamic']:
                if options['rate_continuity']:
                    val = self.rate_jac_templates[control_name]
                    partials['defect_control_rates:{0}_rate'.format(control_name),
                             'control_rates:{0}_rate'.format(control_name)] = \
                        val * dt_dptau

                    end_vals = inputs['control_rates:{0}_rate'.format(control_name)][1:-1:2, ...]
                    start_vals = inputs['control_rates:{0}_rate'.format(control_name)][2:-1:2, ...]

                    partials['defect_control_rates:{0}_rate'.format(control_name),
                             't_duration'] = 0.5 * (start_vals - end_vals)

                if options['rate2_continuity']:
                    val = self.rate_jac_templates[control_name]
                    partials['defect_control_rates:{0}_rate2'.format(control_name),
                             'control_rates:{0}_rate2'.format(control_name)] = \
                        val * dt_dptau**2

                    end_vals = inputs['control_rates:{0}_rate2'.format(control_name)][1:-1:2, ...]
                    start_vals = inputs['control_rates:{0}_rate2'.format(control_name)][2:-1:2, ...]

                    partials['defect_control_rates:{0}_rate2'.format(control_name),
                             't_duration'] = (start_vals - end_vals) * dt_dptau
