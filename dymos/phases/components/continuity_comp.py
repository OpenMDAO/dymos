from __future__ import print_function, division

import numpy as np
from openmdao.api import ExplicitComponent
from six import iteritems, string_types

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units


class ContinuityCompBase(ExplicitComponent):
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

    def _setup_state_continuity(self):
        state_options = self.options['state_options']
        num_segend_nodes = self.options['grid_data'].subset_num_nodes['segment_ends']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1 or compressed:
            return

        for state_name, options in iteritems(state_options):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            self.name_maps[state_name] = {}

            self.name_maps[state_name]['value_names'] = \
                ('states:{0}'.format(state_name),
                 'defect_states:{0}'.format(state_name))

            self.add_input(name='states:{0}'.format(state_name),
                           shape=(num_segend_nodes,) + shape,
                           desc='Values of state {0} at discretization nodes'.format(
                               state_name),
                           units=units)

            self.add_output(
                name='defect_states:{0}'.format(state_name),
                shape=(num_segments - 1,) + shape,
                desc='Consistency constraint values for state {0}'.format(state_name),
                units=units)

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
                'defect_states:{0}'.format(state_name),
                'states:{0}'.format(state_name),
                val=vals, rows=rs, cols=cs,
            )

    def _setup_control_continuity(self):
        control_options = self.options['control_options']
        num_segend_nodes = self.options['grid_data'].subset_num_nodes['segment_ends']
        num_segments = self.options['grid_data'].num_segments
        time_units = self.options['time_units']

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        self.add_input('t_duration', units=time_units, val=1.0,
                       desc='time duration of the phase')

        for control_name, options in iteritems(control_options):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            rate_units = get_rate_units(units, time_units, deriv=1)
            rate2_units = get_rate_units(units, time_units, deriv=2)

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

            #
            # Setup value continuity
            #
            self.name_maps[control_name] = {}

            self.name_maps[control_name]['value_names'] = \
                ('controls:{0}'.format(control_name),
                 'defect_controls:{0}'.format(control_name))

            self.name_maps[control_name]['rate_names'] = \
                ('control_rates:{0}_rate'.format(control_name),
                 'defect_control_rates:{0}_rate'.format(control_name))

            self.name_maps[control_name]['rate2_names'] = \
                ('control_rates:{0}_rate2'.format(control_name),
                 'defect_control_rates:{0}_rate2'.format(control_name))

            self.add_input(
                name='controls:{0}'.format(control_name),
                shape=(num_segend_nodes,) + shape,
                desc='Values of control {0} at discretization nodes'.format(control_name),
                units=units)

            self.add_output(
                name='defect_controls:{0}'.format(control_name),
                val=5*np.ones((num_segments - 1,) + shape),
                desc='Continuity constraint values for control {0}'.format(control_name),
                units=units)

            self.declare_partials(
                'defect_controls:{0}'.format(control_name),
                'controls:{0}'.format(control_name),
                val=vals, rows=rs, cols=cs,
            )

            #
            # Setup first derivative continuity
            #

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

            self.declare_partials(
                'defect_control_rates:{0}_rate'.format(control_name),
                'control_rates:{0}_rate'.format(control_name),
                rows=rs, cols=cs,
            )

            self.declare_partials(
                'defect_control_rates:{0}_rate'.format(control_name),
                't_duration', dependent=True,
            )

            #
            # Setup second derivative continuity
            #

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
        self.name_maps = {}

        self._setup_state_continuity()
        self._setup_control_continuity()

    def _compute_state_continuity(self, inputs, outputs):
        state_options = self.options['state_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1 or compressed:
            return

        for state_name, options in iteritems(state_options):
            input_name, output_name = self.name_maps[state_name]['value_names']
            end_vals = inputs[input_name][1:-1:2, ...]
            start_vals = inputs[input_name][2:-1:2, ...]
            outputs[output_name] = start_vals - end_vals

    def _compute_control_continuity(self, inputs, outputs):
        control_options = self.options['control_options']

        dt_dptau = inputs['t_duration'] / 2.0

        for name, options in iteritems(control_options):
            input_name, output_name = self.name_maps[name]['value_names']
            end_vals = inputs[input_name][1:-1:2, ...]
            start_vals = inputs[input_name][2:-1:2, ...]
            outputs[output_name] = start_vals - end_vals

            input_name, output_name = self.name_maps[name]['rate_names']
            end_vals = inputs[input_name][1:-1:2, ...]
            start_vals = inputs[input_name][2:-1:2, ...]
            outputs[output_name] = (start_vals - end_vals) * dt_dptau

            input_name, output_name = self.name_maps[name]['rate2_names']
            end_vals = inputs[input_name][1:-1:2, ...]
            start_vals = inputs[input_name][2:-1:2, ...]
            outputs[output_name] = (start_vals - end_vals) * dt_dptau ** 2

    def compute(self, inputs, outputs):
        self._compute_state_continuity(inputs, outputs)
        self._compute_control_continuity(inputs, outputs)

    def compute_partials(self, inputs, partials):

        control_options = self.options['control_options']
        dt_dptau = 0.5 * inputs['t_duration']

        for control_name, options in iteritems(control_options):
            input_name, output_name = self.name_maps[control_name]['rate_names']
            val = self.rate_jac_templates[control_name]
            partials[output_name, input_name] = val * dt_dptau

            end_vals = inputs[input_name][1:-1:2, ...]
            start_vals = inputs[input_name][2:-1:2, ...]

            partials[output_name, 't_duration'] = 0.5 * (start_vals - end_vals)

            input_name, output_name = self.name_maps[control_name]['rate2_names']
            val = self.rate_jac_templates[control_name]
            partials[output_name, input_name] = val * dt_dptau**2

            end_vals = inputs[input_name][1:-1:2, ...]
            start_vals = inputs[input_name][2:-1:2, ...]

            partials[output_name, 't_duration'] = (start_vals - end_vals) * dt_dptau


class GaussLobattoContinuityComp(ContinuityCompBase):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.
    """
    def _setup_state_continuity(self):
        state_options = self.options['state_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            return

        super(GaussLobattoContinuityComp, self)._setup_state_continuity()

        for state_name, options in iteritems(state_options):
            if options['continuity'] and not compressed:
                self.add_constraint(name='defect_states:{0}'.format(state_name),
                                    equals=0.0, scaler=1.0, linear=True)

    def _setup_control_continuity(self):
        control_options = self.options['control_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        super(GaussLobattoContinuityComp, self)._setup_control_continuity()

        for control_name, options in iteritems(control_options):

            if options['continuity'] and not compressed:
                self.add_constraint(name='defect_controls:{0}'.format(control_name),
                                    equals=0.0, scaler=1.0, linear=True)

            #
            # Setup first derivative continuity
            #

            if options['rate_continuity']:
                self.add_constraint(name='defect_control_rates:{0}_rate'.format(control_name),
                                    equals=0.0, scaler=options['rate_continuity_scaler'],
                                    linear=False)

            #
            # Setup second derivative continuity
            #

            if options['rate2_continuity']:
                self.add_constraint(name='defect_control_rates:{0}_rate2'.format(control_name),
                                    equals=0.0, scaler=options['rate2_continuity_scaler'],
                                    linear=False)


class RadauPSContinuityComp(ContinuityCompBase):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.
    """
    def _setup_state_continuity(self):
        state_options = self.options['state_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            return

        super(RadauPSContinuityComp, self)._setup_state_continuity()

        for state_name, options in iteritems(state_options):
            if options['continuity'] and not compressed:
                self.add_constraint(name='defect_states:{0}'.format(state_name),
                                    equals=0.0, scaler=1.0, linear=True)

    def _setup_control_continuity(self):
        control_options = self.options['control_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        super(RadauPSContinuityComp, self)._setup_control_continuity()

        for control_name, options in iteritems(control_options):
            if options['continuity']:
                self.add_constraint(name='defect_controls:{0}'.format(control_name),
                                    equals=0.0, scaler=1.0, linear=False)

            #
            # Setup first derivative continuity
            #

            if options['rate_continuity']:
                self.add_constraint(name='defect_control_rates:{0}_rate'.format(control_name),
                                    equals=0.0, scaler=options['rate_continuity_scaler'],
                                    linear=False)

            #
            # Setup second derivative continuity
            #

            if options['rate2_continuity']:
                self.add_constraint(name='defect_control_rates:{0}_rate2'.format(control_name),
                                    equals=0.0, scaler=options['rate2_continuity_scaler'],
                                    linear=False)


class ExplicitContinuityComp(ContinuityCompBase):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.
    """
    def initialize(self):
        super(ExplicitContinuityComp, self).initialize()
        self.options.declare('shooting', default='single', values=('single', 'multiple', 'hybrid'),
                             desc='The shooting method used to integrate across the phase.  Single'
                                  'shooting propagates the state from segment to segment, '
                                  'serially.  Multiple shooting runs each segment in parallel and'
                                  'uses an optimizer to enforce state continuity at segment bounds.'
                                  ' Hybrid propagates the segments in parallel but enforces state '
                                  'continuity with a nonlinear solver.')

    def _setup_state_continuity(self):
        state_options = self.options['state_options']
        gd = self.options['grid_data']
        num_segments = gd.num_segments
        compressed = gd.compressed
        self.state_vars = {}
        self._is_multiple_shooting = self.options['shooting'] == 'multiple'

        if not self._is_multiple_shooting or num_segments <= 1:
            return

        for state_name, options in iteritems(state_options):
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']

            self.state_vars[state_name] = {}
            self.state_vars[state_name]['values'] = []
            self.state_vars[state_name]['defect'] = 'defect_states:{0}'.format(state_name)

            self.add_output(
                name=self.state_vars[state_name]['defect'],
                shape=(num_segments - 1,) + shape,
                desc='Consistency constraint values for state {0}'.format(state_name),
                units=units)

            for iseg in range(num_segments):
                num_steps = gd.num_steps_per_segment[iseg]

                self.state_vars[state_name]['values'].append(
                    'seg_{0}_states:{1}'.format(iseg, state_name))

                self.add_input(name=self.state_vars[state_name]['values'][-1],
                               shape=(num_steps + 1,) + shape,
                               desc='Step values of state {0} in segment {1}'.format(state_name,
                                                                                     iseg - 1),
                               units=units)

                # TODO: Partials currently limited to scalar states
                if iseg == 0:
                    # First segment
                    r = [0]
                    c = [num_steps]
                    val = -1.0
                elif iseg == num_segments - 1:
                    # Last segment
                    r = [num_segments - 2]
                    c = [0]
                    val = 1.0
                else:
                    #
                    r = [iseg - 1, iseg]
                    c = [0, num_steps]
                    val = [1.0, -1.0]

                self.declare_partials(of=self.state_vars[state_name]['defect'],
                                      wrt=self.state_vars[state_name]['values'][-1],
                                      rows=r, cols=c, val=val)

            # Add the constraint
            if options['continuity']:
                self.add_constraint(name='defect_states:{0}'.format(state_name),
                                    equals=0.0, scaler=1.0, linear=False)

    def _setup_control_continuity(self):
        control_options = self.options['control_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        super(ExplicitContinuityComp, self)._setup_control_continuity()

        for control_name, options in iteritems(control_options):

            if options['continuity'] and not compressed:
                self.add_constraint(name='defect_controls:{0}'.format(control_name),
                                    equals=0.0, scaler=1.0, linear=True)

            #
            # Setup first derivative continuity
            #

            if options['rate_continuity']:
                self.add_constraint(name='defect_control_rates:{0}_rate'.format(control_name),
                                    equals=0.0, scaler=options['rate_continuity_scaler'],
                                    linear=False)

            #
            # Setup second derivative continuity
            #

            if options['rate2_continuity']:
                self.add_constraint(name='defect_control_rates:{0}_rate2'.format(control_name),
                                    equals=0.0, scaler=options['rate2_continuity_scaler'],
                                    linear=False)

    def _compute_state_continuity(self, inputs, outputs):
        state_options = self.options['state_options']
        num_segments = self.options['grid_data'].num_segments

        if not self._is_multiple_shooting:
            return

        for i in range(1, num_segments):
            for state_name, options in iteritems(state_options):
                left = inputs[self.state_vars[state_name]['values'][i-1]][-1, ...]
                right = inputs[self.state_vars[state_name]['values'][i]][0, ...]
                defect = outputs[self.state_vars[state_name]['defect']]
                defect[i - 1, ...] = right - left
