from ...common.continuity_comp import ContinuityCompBase


class RungeKuttaControlContinuityComp(ContinuityCompBase):
    """
    ContinuityComp defines constraints to ensure continuity between adjacent segments.
    """
    def _configure_state_continuity(self):
        pass

    def _compute_state_continuity(self, inputs, outputs):
        pass

    def _configure_control_continuity(self):
        control_options = self.options['control_options']
        num_segments = self.options['grid_data'].num_segments
        compressed = self.options['grid_data'].compressed

        if num_segments <= 1:
            # Control rate continuity is enforced even with compressed transcription
            return

        super(RungeKuttaControlContinuityComp, self)._configure_control_continuity()

        for control_name, options in control_options.items():
            if not compressed and options['continuity']:
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
