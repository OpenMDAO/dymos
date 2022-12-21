from collections.abc import Sequence

import numpy as np

from .phase import Phase
from ..transcriptions import DirectShooting
from ..utils.misc import _unspecified


class DirectShootingPhase(Phase):
    """
    The DirectShootingPhase object in Dymos.

    The DirectShootingPhase object in dymos inherits from PhaseBase but is used to override some base methods with ones
    that will warn about certain options or methods being invalid for the DirectShootingPhase.

    Parameters
    ----------
    from_phase : <Phase> or None
        A phase instance from which the initialized phase should copy its data.
    **kwargs : dict
        Dictionary of optional phase arguments.
    """

    def __init__(self, from_phase=None, **kwargs):
        super().__init__(from_phase=from_phase, **kwargs)
        self.simulate_options = None

    def initialize(self):
        """
        Declare instantiation options for the phase.
        """
        super().initialize()
        self.options.declare('grid',
                             desc='The type of grid used to provide the control input nodes and output nodes.')
        self.options.declare('times_per_seg', types=int, allow_none=True, default=None,
                             desc='Number of output times in each segment of the explicit integration. If None,'
                                  'use the output nodes of the transcription.')
        self.options.declare('num_segments', types=int, desc='Number of segments')
        self.options.declare('segment_ends', default=None, types=(Sequence, np.ndarray),
                             allow_none=True, desc='Locations of segment ends or None for equally '
                             'spaced segments')
        self.options.declare('order', default=3, types=(int, Sequence, np.ndarray),
                             desc='Order of the state transcription. The order of the control '
                                  'transcription is `order - 1`.')
        self.options.declare('compressed', default=True, types=bool,
                             desc='Use compressed transcription, meaning state and control values'
                                  'at segment boundaries are not duplicated on input.  This '
                                  'implicitly enforces value continuity between segments but in '
                                  'some cases may make the problem more difficult to solve.')


    def setup(self):
        """
        Build the model hierarchy for a Dymos DirectShootingPhase.
        """
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        
        self.options.declare('method', default='DOP853', desc='The integration method used.')
        self.options.declare('atol', types=float, default=1.0E-6)
        self.options.declare('rtol', types=float, default=1.0E-9)
        self.options.declare('first_step', types=float, allow_none=True, default=None)
        self.options.declare('max_step', types=float, default=np.inf)
        self.options.declare('times_per_seg', types=(int,), allow_none=True, default=None,
                             desc='The number of output times per segment. If specified, they are evenly spaced. If '
                                  'not specified, output at all nodes in the segment as given by the transcription.')
        self.options.declare('propagate_derivs', types=bool, default=True,
                             desc='If True, propagate the state and derivatives of the state and time with respect to '
                                  'the integration parameters. If False, only propagate the primal states. If only '
                                  'using this transcription to propagate an ODE and derivatives are needed, setting '
                                  'this option to False should result in faster execution.')
        self.options.declare('subprob_reports', default=False,
                             desc='Controls the reports made when running the subproblems for DirectShooting')
        self.options.declare('input_grid', values=['radau-ps', 'gauss-lobatto'],
                             default='gauss-lobatto', desc='The grid distribution used to layout the control inputs.')
        self.options.declare('input_grid_order', types=int, default=3,
                             desc='The order used to determine the number of nodes in the control inputs. For '
                                  'consistency with other transcriptions, controls are assumed to be polynomials of '
                                  'input_grid_order - 1.')
        self.options.declare('output_grid', values=['radau-ps', 'gauss-lobatto', 'uniform'], allow_none=True,
                             default=None,
                             desc='The grid distribution determining the location of the output nodes. If rate '
                                  'constraints are being imposed on outputs, then "uniform" should not be used to '
                                  'avoid interpolation issues. The default value of None will result in the use of the '
                                  'input_grid for outputs. This is useful for the implementation of path constraints '
                                  'but can result in highly nonlinear dynamics being smoothed over in the outputs. '
                                  'When used for validation through simulation, it is generally wise to choose an '
                                  'output grid that is more dense than the input grid to capture this nonlinearity.')
        self.options.declare('output_grid_order', types=int, allow_none=True, default=None,
                             desc='The order of the output grid, affecting the number of nodes in the output. If None, '
                                  'use the same order as the input grid. See the description for output_grid, as '
                                  'the notes there apply to this option as well.')
        
        tx = self.options['transcription'] = DirectShooting(method=self.options['method'],
                                                            num_segments=self.options['num_segments'],
                                                            atol=self.options['atol'],
                                                            rtol=self.options['rtol'],
                                                            first_step=self.options['first_step'],
                                                            max_step=self.options['max_step'],
                                                            times_per_seg=self.options['times_per_seg'],
                                                            propagate_derivs=self.options['propagate_derivs'],
                                                            subprob_reports=self.options['subprob_reports'],
                                                            input_grid=self.options['input_grid'],
                                                            input_grid_order=self.options['input_grid_order'],
                                                            output_grid=self.options['output_grid'],
                                                            output_grid_order=self.options['output_grid_order'])
        tx.setup_time(self)

        if self.control_options:
            tx.setup_controls(self)

        if self.polynomial_control_options:
            tx.setup_polynomial_controls(self)

        if self.parameter_options:
            tx.setup_parameters(self)

        tx.setup_states(self)
        self._check_ode()
        tx.setup_ode(self)

        tx.setup_timeseries_outputs(self)
        tx.setup_defects(self)
        tx.setup_solvers(self)

    def simulate(self, times_per_seg=10, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                 first_step=_unspecified, max_step=_unspecified, record_file=None):
        """
        Stub to make sure users are informed that simulate cannot be done on DirectShootingPhase.

        Parameters
        ----------
        times_per_seg : int or None
            Number of equally spaced times per segment at which output is requested.  If None,
            output will be provided at all Nodes.
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for scipy.integrate.solve_ivp.
        rtol : float
            Relative convergence tolerance for scipy.integrate.solve_ivp.
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.
        record_file : str or None
            If a string, the file to which the result of the simulation will be saved.
            If None, no record of the simulation will be saved.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        raise NotImplementedError('Method `simulate` is not available for DirectShootingPhase.')

    def set_simulate_options(self, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                             first_step=_unspecified, max_step=_unspecified):
        """
        Stub to make sure users are informed that simulate cannot be done on DirectShootingPhase.

        Parameters
        ----------
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for scipy.integrate.solve_ivp.
        rtol : float
            Relative convergence tolerance for scipy.integrate.solve_ivp.
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.
        """
        raise NotImplementedError('Method set_simulate_options is not available for DirectShootingPhase.')
