from __future__ import print_function, division

import numpy as np
from openmdao.api import ExplicitComponent
from six import iteritems, string_types

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units


class ContinuityComp(ExplicitComponent):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.
    """

    def initialize(self):

        self.metadata.declare('grid_data', types=GridData,
                              desc='Container object for grid info')

        self.metadata.declare('state_options', types=dict,
                              desc='Dictionary of state names/options for the phase')

        self.metadata.declare('control_options', types=dict,
                              desc='Dictionary of control names/options for the phase')

        self.metadata.declare('time_units', default=None, allow_none=True, types=string_types,
                              desc='Units of the integration variable')

        self.metadata.declare('enforce_state_continuity', default=True, types=bool,
                              desc='Whether to add state continuity constraints')

    def _setup_value_continuity(self):
        state_options = self.metadata['state_options']
        control_options = self.metadata['control_options']
        segment_indices = self.metadata['grid_data'].subset_segment_indices['state_disc']
        num_disc_nodes = self.metadata['grid_data'].subset_num_nodes['state_disc']
        num_segments = self.metadata['grid_data'].num_segments

        self.jacs = {}

        for state_name, options in iteritems(state_options):
            shape = options['shape']
            units = options['units']

            self.add_input(
                name='states:{0}'.format(state_name),
                shape=(num_disc_nodes,) + shape,
                desc='Values of state {0} at discretization nodes'.format(state_name),
                units=units)

            self.add_output(
                name='defect_states:{0}'.format(state_name),
                shape=(num_segments - 1,) + shape,
                desc='Consistency constraint values for state {0}'.format(state_name),
                units=units)

            if self.metadata['enforce_state_continuity']:
                self.add_constraint(name='defect_states:{0}'.format(state_name),
                                    equals=0.0, scaler=1.0, linear=True)

            size = np.prod(shape)

            vals = np.zeros((num_segments - 1, 2, size))
            rows = np.zeros((num_segments - 1, 2, size), int)
            cols = np.zeros((num_segments - 1, 2, size), int)

            arange = np.arange(size)

            vals[:, 0] = 1.0
            vals[:, 1] = -1.0
            for iseg in range(num_segments - 1):
                rows[iseg, :, :] = iseg * size + arange
                cols[iseg, 0, :] = (segment_indices[iseg, 1] - 1) * size + arange
                cols[iseg, 1, :] = segment_indices[iseg + 1, 0] * size + arange

            self.jacs[state_name] = (vals.ravel(), rows.ravel(), cols.ravel())

        for control_name, options in iteritems(control_options):
            shape = options['shape']
            units = options['units']

            if options['dynamic'] and options['continuity']:
                self.add_input(
                    name='controls:{0}'.format(control_name),
                    shape=(num_disc_nodes,) + shape,
                    desc='Values of control {0} at discretization nodes'.format(control_name),
                    units=units)

                self.add_output(
                    name='defect_controls:{0}'.format(control_name),
                    shape=(num_segments - 1,) + shape,
                    desc='Consistency constraint values for control {0}'.format(control_name),
                    units=units)

                self.add_constraint(name='defect_controls:{0}'.format(control_name),
                                    equals=0.0, scaler=1.0, linear=True)

                size = np.prod(shape)

                vals = np.zeros((num_segments - 1, 2, size))
                rows = np.zeros((num_segments - 1, 2, size), int)
                cols = np.zeros((num_segments - 1, 2, size), int)

                arange = np.arange(size)

                vals[:, 0] = 1.0
                vals[:, 1] = -1.0
                for iseg in range(num_segments - 1):
                    rows[iseg, :, :] = iseg * size + arange
                    cols[iseg, 0, :] = (segment_indices[iseg, 1] - 1) * size + arange
                    cols[iseg, 1, :] = segment_indices[iseg + 1, 0] * size + arange

                self.jacs[control_name] = (vals.ravel(), rows.ravel(), cols.ravel())

        for state_name, options in iteritems(state_options):
            vals, rows, cols = self.jacs[state_name]
            self.declare_partials(
                'defect_states:{0}'.format(state_name),
                'states:{0}'.format(state_name),
                val=vals, rows=rows, cols=cols,
            )

        for control_name, options in iteritems(control_options):
            if options['dynamic'] and options['continuity']:
                vals, rows, cols = self.jacs[control_name]
                self.declare_partials(
                    'defect_controls:{0}'.format(control_name),
                    'controls:{0}'.format(control_name),
                    val=vals, rows=rows, cols=cols,
                )

    def _setup_rate_continuity(self):
        control_options = self.metadata['control_options']
        segment_indices = self.metadata['grid_data'].subset_segment_indices['disc']
        num_disc_nodes = self.metadata['grid_data'].subset_num_nodes['disc']
        num_segments = self.metadata['grid_data'].num_segments

        for control_name, options in iteritems(control_options):
            shape = options['shape']
            units = options['units']
            rate_units = get_rate_units(units, self.metadata['time_units'], deriv=1)
            rate2_units = get_rate_units(units, self.metadata['time_units'], deriv=2)

            if options['opt'] and options['dynamic']:

                size = np.prod(shape)

                vals = np.zeros((num_segments - 1, 2, size))
                rows = np.zeros((num_segments - 1, 2, size), int)
                cols = np.zeros((num_segments - 1, 2, size), int)

                arange = np.arange(size)

                vals[:, 0] = 1.0
                vals[:, 1] = -1.0
                for iseg in range(num_segments - 1):
                    rows[iseg, :, :] = iseg * size + arange
                    cols[iseg, 0, :] = (segment_indices[iseg, 1] - 1) * size + arange
                    cols[iseg, 1, :] = segment_indices[iseg + 1, 0] * size + arange

                self.jacs[control_name] = (vals.ravel(), rows.ravel(), cols.ravel())

                if options['rate_continuity']:
                    self.add_input(
                        name='control_rates:{0}_rate'.format(control_name),
                        shape=(num_disc_nodes,) + shape,
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
                                        equals=0.0, scaler=1.0, linear=False)

                    vals, rows, cols = self.jacs[control_name]
                    self.declare_partials(
                        'defect_control_rates:{0}_rate'.format(control_name),
                        'control_rates:{0}_rate'.format(control_name),
                        val=vals, rows=rows, cols=cols,
                    )

                if options['rate2_continuity']:
                    self.add_input(
                        name='control_rates:{0}_rate2'.format(control_name),
                        shape=(num_disc_nodes,) + shape,
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
                                        equals=0.0, scaler=1.0, linear=False)

                    vals, rows, cols = self.jacs[control_name]
                    self.declare_partials(
                        'defect_control_rates:{0}_rate2'.format(control_name),
                        'control_rates:{0}_rate2'.format(control_name),
                        val=vals, rows=rows, cols=cols,
                    )

    def setup(self):
        self.jacs = {}
        compressed = self.metadata['grid_data'].compressed
        if not compressed:
            self._setup_value_continuity()
        self._setup_rate_continuity()

    def compute(self, inputs, outputs):
        state_options = self.metadata['state_options']
        control_options = self.metadata['control_options']
        compressed = self.metadata['grid_data'].compressed

        if not compressed:
            for state_name, options in iteritems(state_options):
                vals, rows, cols = self.jacs[state_name]

                outputs_raw = outputs['defect_states:{0}'.format(state_name)]
                inputs_raw = inputs['states:{0}'.format(state_name)]

                outputs_flat = outputs_raw.reshape((np.prod(outputs_raw.shape)))
                inputs_flat = inputs_raw.reshape((np.prod(inputs_raw.shape)))

                outputs_flat[:] = 0.
                np.add.at(outputs_flat, rows, vals * inputs_flat[cols])

            for control_name, options in iteritems(control_options):
                if options['dynamic'] and options['continuity']:
                    vals, rows, cols = self.jacs[control_name]

                    outputs_raw = outputs['defect_controls:{0}'.format(control_name)]
                    inputs_raw = inputs['controls:{0}'.format(control_name)]

                    outputs_flat = outputs_raw.reshape((np.prod(outputs_raw.shape)))
                    inputs_flat = inputs_raw.reshape((np.prod(inputs_raw.shape)))

                    outputs_flat[:] = 0.
                    np.add.at(outputs_flat, rows, vals * inputs_flat[cols])

        for control_name, options in iteritems(control_options):
            if options['opt'] and options['dynamic']:
                if options['rate_continuity']:
                    vals, rows, cols = self.jacs[control_name]

                    outputs_raw = outputs['defect_control_rates:{0}_rate'.format(control_name)]
                    inputs_raw = inputs['control_rates:{0}_rate'.format(control_name)]

                    outputs_flat = outputs_raw.reshape((np.prod(outputs_raw.shape)))
                    inputs_flat = inputs_raw.reshape((np.prod(inputs_raw.shape)))

                    outputs_flat[:] = 0.
                    np.add.at(outputs_flat, rows, vals * inputs_flat[cols])

                if options['rate2_continuity']:
                    vals, rows, cols = self.jacs[control_name]

                    outputs_raw = outputs['defect_control_rates:{0}_rate2'.format(control_name)]
                    inputs_raw = inputs['control_rates:{0}_rate2'.format(control_name)]

                    outputs_flat = outputs_raw.reshape((np.prod(outputs_raw.shape)))
                    inputs_flat = inputs_raw.reshape((np.prod(inputs_raw.shape)))

                    outputs_flat[:] = 0.
                    np.add.at(outputs_flat, rows, vals * inputs_flat[cols])
