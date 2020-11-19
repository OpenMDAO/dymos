import numpy as np
import openmdao.api as om

from ..grid_data import GridData
from ...utils.misc import get_rate_units
from ...options import options as dymos_options


class ContinuityCompBase(om.ExplicitComponent):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):

        self.options.declare('grid_data', types=GridData,
                             desc='Container object for grid info')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the phase')

        self.options.declare('control_options', types=dict,
                             desc='Dictionary of control names/options for the phase')

        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of the integration variable')

    def _configure_state_continuity(self):
        state_options = self.options['state_options']
        num_segend_nodes = self.options['grid_data'].subset_num_nodes['segment_ends']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1 or compressed:
            return

        for state_name, options in state_options.items():
            shape = options['shape'] if options['shape'] is not None else (1, )
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

    def _configure_control_continuity(self):
        control_options = self.options['control_options']
        num_segend_nodes = self.options['grid_data'].subset_num_nodes['segment_ends']
        num_segments = self.options['grid_data'].num_segments
        time_units = self.options['time_units']

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        self.add_input('t_duration', units=time_units, val=1.0,
                       desc='time duration of the phase')

        for control_name, options in control_options.items():
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

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        self.rate_jac_templates = {}
        self.name_maps = {}

        self._configure_state_continuity()
        self._configure_control_continuity()

    def _compute_state_continuity(self, inputs, outputs):
        state_options = self.options['state_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1 or compressed:
            return

        for state_name, options in state_options.items():
            input_name, output_name = self.name_maps[state_name]['value_names']
            end_vals = inputs[input_name][1:-1:2, ...]
            start_vals = inputs[input_name][2:-1:2, ...]
            outputs[output_name] = start_vals - end_vals

    def _compute_control_continuity(self, inputs, outputs):
        control_options = self.options['control_options']

        dt_dptau = inputs['t_duration'] / 2.0

        for name, options in control_options.items():
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

        for control_name, options in control_options.items():
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
    def _configure_state_continuity(self):
        state_options = self.options['state_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            return

        super(GaussLobattoContinuityComp, self)._configure_state_continuity()

        for state_name, options in state_options.items():
            if options['continuity'] and not compressed:

                # linear if states are optimized, because they are dvs.
                # but nonlinear if solve_segments, because its like multiple shooting
                is_linear = not options['solve_segments']
                self.add_constraint(name='defect_states:{0}'.format(state_name),
                                    equals=0.0, scaler=1.0, linear=is_linear)

    def _configure_control_continuity(self):
        control_options = self.options['control_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        super(GaussLobattoContinuityComp, self)._configure_control_continuity()

        for control_name, options in control_options.items():

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
    def _configure_state_continuity(self):
        state_options = self.options['state_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            return

        super(RadauPSContinuityComp, self)._configure_state_continuity()

        for state_name, options in state_options.items():
            if options['continuity'] and not compressed:
                # linear if states are optimized, because they are dvs.
                # but nonlinear if solve_segments, because its like multiple shooting
                is_linear = not options['solve_segments']

                self.add_constraint(name='defect_states:{0}'.format(state_name),
                                    equals=0.0, scaler=1.0, linear=is_linear)

    def _configure_control_continuity(self):
        control_options = self.options['control_options']
        num_segments = self.options['grid_data'].num_segments

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        super(RadauPSContinuityComp, self)._configure_control_continuity()

        for control_name, options in control_options.items():
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
