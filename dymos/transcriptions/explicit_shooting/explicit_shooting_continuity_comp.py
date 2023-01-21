import numpy as np

from ..grid_data import GaussLobattoGrid
from ...utils.misc import get_rate_units

from ..common.continuity_comp import ContinuityCompBase


class ExplicitShootingContinuityComp(ContinuityCompBase):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._controls_to_enforce = set()
        self._control_rates_to_enforce = set()
        self._control_rates2_to_enforce = set()

        self.rate_jac_templates = {}
        self.name_maps = {}

    def _configure_state_continuity(self):
        # TODO This method will be used when multiple shooting is implemented.
        pass

        # state_options = self.options['state_options']
        # num_segments = self.options['grid_data'].num_segments
        # compressed = self.options['grid_data'].compressed
        #
        # if num_segments <= 1:
        #     return
        #
        # super(ExplicitShootingContinuityComp, self)._configure_state_continuity()
        #
        # for state_name, options in state_options.items():
        #     if options['continuity'] and not compressed:
        #         # State continuity is nonlinear in (TBD) explicit multiple shooting phases
        #         self.add_constraint(name=f'defect_states:{state_name}',
        #                             equals=0.0, scaler=1.0, linear=False)

    def _configure_control_continuity(self, controls_to_enforce=None, control_rates_to_enforce=None,
                                      control_rates2_to_enforce=None):
        """
        Configures control continuity.

        Each argument contains the names of those variables which require continuity.

        Parameters
        ----------
        controls : set or Sequence of str or None
            The names of the controls whose values are to be enforced at segment boundaries.
        control_rates : set or Sequence of str or None
            The names of controls whose rates are to be enforced at segment boundaries.
        control_rates2 : set or Sequence of str or None
            The names of controls whose second derivatives are to be enforced at the segment boundaries.
        """

        control_options = self.options['control_options']
        num_segend_nodes = self.options['grid_data'].subset_num_nodes['segment_ends']
        num_segments = self.options['grid_data'].num_segments
        time_units = self.options['time_units']

        self._controls_to_enforce = controls_to_enforce
        self._control_rates_to_enforce = control_rates_to_enforce
        self._control_rates2_to_enforce = control_rates2_to_enforce

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        any_rate_continuity = False

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

            if control_name in controls_to_enforce:
                self.name_maps[control_name]['value_names'] = \
                    (f'controls:{control_name}',
                     f'defect_controls:{control_name}')

                self.add_input(
                    name=f'controls:{control_name}',
                    shape=(num_segend_nodes,) + shape,
                    desc=f'Values of control {control_name} at segment endpoint nodes',
                    units=units)

                self.add_output(
                    name=f'defect_controls:{control_name}',
                    val=5*np.ones((num_segments - 1,) + shape),
                    desc=f'Continuity constraint values for control {control_name}',
                    units=units)

                self.declare_partials(
                    f'defect_controls:{control_name}',
                    f'controls:{control_name}',
                    val=vals, rows=rs, cols=cs,
                )

                linear_cnty = isinstance(self.options['grid_data'], GaussLobattoGrid)
                self.add_constraint(name=f'defect_controls:{control_name}',
                                    equals=0.0, scaler=1.0, linear=linear_cnty)

            if control_name in control_rates_to_enforce:
                any_rate_continuity = True

                self.name_maps[control_name]['rate_names'] = \
                    (f'control_rates:{control_name}_rate',
                     f'defect_control_rates:{control_name}_rate')

                self.add_input(
                    name=f'control_rates:{control_name}_rate',
                    shape=(num_segend_nodes,) + shape,
                    desc=f'Values of control {control_name} derivative at segment endpoint nodes',
                    units=rate_units)

                self.add_output(
                    name=f'defect_control_rates:{control_name}_rate',
                    shape=(num_segments - 1,) + shape,
                    desc=f'Consistency constraint values for control {control_name} derivative',
                    units=rate_units)

                self.declare_partials(
                    f'defect_control_rates:{control_name}_rate',
                    f'control_rates:{control_name}_rate',
                    rows=rs, cols=cs,
                )

                self.declare_partials(
                    f'defect_control_rates:{control_name}_rate',
                    't_duration', dependent=True,
                )

                self.add_constraint(name=f'defect_control_rates:{control_name}_rate',
                                    equals=0.0, scaler=options['rate_continuity_scaler'],
                                    linear=False)

            if control_name in control_rates2_to_enforce:
                any_rate_continuity = True

                self.name_maps[control_name]['rate2_names'] = \
                    (f'control_rates:{control_name}_rate2',
                     f'defect_control_rates:{control_name}_rate2')

                self.add_input(
                    name=f'control_rates:{control_name}_rate2',
                    shape=(num_segend_nodes,) + shape,
                    desc=f'Values of control {control_name} second derivative '
                         'at discretization nodes',
                    units=rate2_units)

                self.add_output(
                    name=f'defect_control_rates:{control_name}_rate2',
                    shape=(num_segments - 1,) + shape,
                    desc='Consistency constraint values for control '
                         f'{control_name} second derivative',
                    units=rate2_units)

                self.declare_partials(
                    f'defect_control_rates:{control_name}_rate2',
                    f'control_rates:{control_name}_rate2',
                    rows=rs, cols=cs,
                )

                self.declare_partials(
                    f'defect_control_rates:{control_name}_rate2',
                    't_duration', dependent=True
                )

                self.add_constraint(name=f'defect_control_rates:{control_name}_rate2',
                                    equals=0.0, scaler=options['rate2_continuity_scaler'],
                                    linear=False)

        if any_rate_continuity:
            self.add_input('t_duration', units=time_units, val=1.0, desc='time duration of the phase')

    def configure_io(self, controls_to_enforce=None, control_rates_to_enforce=None,
                     control_rates2_to_enforce=None):
        """
        Configures control continuity.

        Each argument contains the names of those variables which require continuity.

        Parameters
        ----------
        controls_to_enforce : set or Sequence of str or None
            The names of the controls whose values are to be enforced at segment boundaries.
        control_rates_to_enforce : set or Sequence of str or None
            The names of controls whose rates are to be enforced at segment boundaries.
        control_rates2_to_enforce : set or Sequence of str or None
            The names of controls whose second derivatives are to be enforced at the segment boundaries.
        """
        self.rate_jac_templates = {}
        self.name_maps = {}

        self._configure_control_continuity(controls_to_enforce=controls_to_enforce,
                                           control_rates_to_enforce=control_rates_to_enforce,
                                           control_rates2_to_enforce=control_rates2_to_enforce)
        self._configure_state_continuity()

    def _compute_state_continuity(self, inputs, outputs):
        pass

    def _compute_control_continuity(self, inputs, outputs):
        control_options = self.options['control_options']

        if any([options['rate_continuity'] or options['rate2_continuity'] for options in control_options.values()]):
            dt_dptau = inputs['t_duration'] / 2.0

        for name, options in control_options.items():
            if name in self._controls_to_enforce:
                input_name, output_name = self.name_maps[name]['value_names']
                end_vals = inputs[input_name][1:-1:2, ...]
                start_vals = inputs[input_name][2:-1:2, ...]
                outputs[output_name] = start_vals - end_vals

            if name in self._control_rates_to_enforce:
                input_name, output_name = self.name_maps[name]['rate_names']
                end_vals = inputs[input_name][1:-1:2, ...]
                start_vals = inputs[input_name][2:-1:2, ...]
                outputs[output_name] = (start_vals - end_vals) * dt_dptau

            if name in self._control_rates2_to_enforce:
                input_name, output_name = self.name_maps[name]['rate2_names']
                end_vals = inputs[input_name][1:-1:2, ...]
                start_vals = inputs[input_name][2:-1:2, ...]
                outputs[output_name] = (start_vals - end_vals) * dt_dptau ** 2
