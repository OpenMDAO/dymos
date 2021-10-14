from ..common.continuity_comp import ContinuityCompBase


class ExplicitShootingContinuityComp(ContinuityCompBase):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
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

    def _configure_control_continuity(self):
        super()._configure_control_continuity()
        control_options = self.options['control_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed
        grid_tx = self.options['grid_data'].transcription

        if num_segments <= 1:
            # Control value and rate continuity is enforced even with compressed transcription
            return

        for control_name, options in control_options.items():
            if options['continuity'] and not compressed:
                linear_cnty = grid_tx == 'gauss-lobatto'
                self.add_constraint(name=f'defect_controls:{control_name}',
                                    equals=0.0, scaler=1.0, linear=linear_cnty)

            #
            # Setup first derivative continuity
            #

            if options['rate_continuity']:
                self.add_constraint(name=f'defect_control_rates:{control_name}_rate',
                                    equals=0.0, scaler=options['rate_continuity_scaler'],
                                    linear=False)

            #
            # Setup second derivative continuity
            #

            if options['rate2_continuity']:
                self.add_constraint(name=f'defect_control_rates:{control_name}_rate2',
                                    equals=0.0, scaler=options['rate2_continuity_scaler'],
                                    linear=False)

    def _compute_state_continuity(self, inputs, outputs):
        pass
