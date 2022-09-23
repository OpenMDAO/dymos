from .phase import Phase
from ..transcriptions.transcription_base import TranscriptionBase

from ..utils.misc import _unspecified


class AnalyticPhase(Phase):

    def initialize(self):
        """
        Declare instantiation options for the phase.
        """
        self.options.declare('rhs_class', default=None,
                             desc='System providing the outputs of the analytic solution.',
                             recordable=False)
        self.options.declare('rhs_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('grid', values=('radau', 'gauss-lobatto'), default='radau',
                             desc='The type of grid used to define the nodes, one of \'radau\' or \'gauss-lobatto\'.')

    def add_state(self, name, units=_unspecified, shape=_unspecified,
                  rate_source=_unspecified, targets=_unspecified,
                  val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                  lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                  ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                  defect_ref=_unspecified, solve_segments=_unspecified, connected_initial=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support state as special variables. Initial values of '
                                  'states can be provided as parameters.')

    def set_state_options(self, name, units=_unspecified, shape=_unspecified,
                          rate_source=_unspecified, targets=_unspecified,
                          val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                          lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                          ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                          defect_ref=_unspecified, solve_segments=_unspecified, connected_initial=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support state as special variables. Initial values of '
                                  'states can be provided as parameters.')

    def add_control(self, name, units=_unspecified, desc=_unspecified, opt=_unspecified,
                    fix_initial=_unspecified, fix_final=_unspecified, targets=_unspecified,
                    rate_targets=_unspecified, rate2_targets=_unspecified, val=_unspecified,
                    shape=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                    adder=_unspecified, ref0=_unspecified, ref=_unspecified, continuity=_unspecified,
                    continuity_scaler=_unspecified, rate_continuity=_unspecified,
                    rate_continuity_scaler=_unspecified, rate2_continuity=_unspecified,
                    rate2_continuity_scaler=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support controls.')

    def set_control_options(self, name, units=_unspecified, desc=_unspecified, opt=_unspecified,
                            fix_initial=_unspecified, fix_final=_unspecified, targets=_unspecified,
                            rate_targets=_unspecified, rate2_targets=_unspecified, val=_unspecified,
                            shape=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                            adder=_unspecified, ref0=_unspecified, ref=_unspecified, continuity=_unspecified,
                            continuity_scaler=_unspecified, rate_continuity=_unspecified,
                            rate_continuity_scaler=_unspecified, rate2_continuity=_unspecified,
                            rate2_continuity_scaler=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support controls.')

    def add_polynomial_control(self, name, order, desc=_unspecified, val=_unspecified, units=_unspecified,
                               opt=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                               lower=_unspecified, upper=_unspecified,
                               scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                               ref=_unspecified, targets=_unspecified, rate_targets=_unspecified,
                               rate2_targets=_unspecified, shape=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support polynomial controls.')

    def set_polynomial_control_options(self, name, order, desc=_unspecified, val=_unspecified,
                                       units=_unspecified, opt=_unspecified, fix_initial=_unspecified,
                                       fix_final=_unspecified, lower=_unspecified, upper=_unspecified,
                                       scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                                       ref=_unspecified, targets=_unspecified, rate_targets=_unspecified,
                                       rate2_targets=_unspecified, shape=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support polynomial controls.')

    def setup(self):
        """
        Build the model hierarchy for a Dymos AnalyticPhase.
        """
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        transcription = self.options['transcription']
        transcription.setup_time(self)

        if self.control_options:
            transcription.setup_controls(self)

        if self.polynomial_control_options:
            transcription.setup_polynomial_controls(self)

        if self.parameter_options:
            transcription.setup_parameters(self)

        transcription.setup_states(self)
        self._check_ode()
        transcription.setup_ode(self)

        transcription.setup_timeseries_outputs(self)
        transcription.setup_defects(self)
        transcription.setup_solvers(self)
